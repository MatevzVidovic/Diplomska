

import logging
import yaml
import os.path as osp
import python_logger.log_helper as py_log_always_on

with open(f"{osp.join('pylog_configs', 'active_logging_config.txt')}", 'r') as f:
    cfg_name = f.read()
    yaml_path = osp.join('pylog_configs', cfg_name)

log_config_path = osp.join(yaml_path)
do_log = False
if osp.exists(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
        file_log_setting = config.get(osp.basename(__file__), False)
        if file_log_setting:
            do_log = True

print(f"{osp.basename(__file__)} do_log: {do_log}")
if do_log:
    import python_logger.log_helper as py_log
else:
    import python_logger.log_helper_off as py_log

MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)


import y_helpers.shared as shared

import os
import pickle

import torch
from torch.utils.data import DataLoader


import y_helpers.json_handler as jh
from y_helpers.img_and_fig_tools import show_image, save_plt_fig, smart_conversion, save_img
from y_helpers.model_eval_graphs import resource_graph, show_results

from y_framework.model_wrapper import ModelWrapper

from y_framework.log_handlers import TrainingLogs, PruningLogs, log_flops_and_weights















def viscinity_save_check(viscinity_save_params, curr_resource_percentage):

    viscinity_save_percentages, viscinity_save_margin = viscinity_save_params["resource_percentage_list"], viscinity_save_params["margin_for_save"]
    for vsp in viscinity_save_percentages:
        if curr_resource_percentage >= vsp - viscinity_save_margin and curr_resource_percentage <= vsp + viscinity_save_margin:
            return True
    return False



@py_log.autolog(passed_logger=MY_LOGGER)
def perform_save(model_wrapper: ModelWrapper, training_logs: TrainingLogs, pruning_logs: PruningLogs, train_iter, unique_id, val_error=None, test_error=None, train_error=None):

    new_model_filename, _ = model_wrapper.save(f"{train_iter}_{unique_id}")
    pruning_logs.confirm_last_pruning_train_iter()

    if val_error is None or test_error is None:
        # This happens if we do a manual save or pruning with keyboard input, and so no training happened in this run yet.
        new_log = None

        if training_logs.last_log is not None:
            # If there were other runs before and training happened, then we are saving the last model that was trained, and so we can repeat the log.
            v = training_logs.last_log["val_err"]
            t = training_logs.last_log["test_err"]
            te = training_logs.last_log["train_err"] if "train_err" in training_logs.last_log else None
            ti = training_logs.last_log["train_iter"]
            # new_log = (v, t, ti, new_model_filename, unique_id, True)
            new_log = ({"val_err": v, "test_err": t, "train_err": te, "train_iter": ti, "model_filename": new_model_filename, "unique_id": unique_id, "is_not_automatic": True})

        training_logs.add_log(new_log)
    else:
        new_log = {"val_err": val_error, "test_err": test_error, "train_err": train_error, "train_iter": train_iter, "model_filename": new_model_filename, "unique_id": unique_id, "is_not_automatic": False}
        training_logs.add_log(new_log)


    training_logs.pickle_training_logs(train_iter, unique_id)
    pruning_logs.pickle_pruning_logs(train_iter, unique_id)
    

    return training_logs, pruning_logs






@py_log.autolog(passed_logger=MY_LOGGER)
def train_automatically(model_wrapper: ModelWrapper, main_save_path, val_stop_fn=None, max_training_iters=1e9, max_total_training_iters=1e9,
                        max_auto_prunings=1e9, train_iter_possible_stop=5, pruning_phase=False, cleaning_err_key="loss", 
                        cleanup_k=3, num_of_epochs_per_training=1, pruning_kwargs_dict=None, viscinity_save_params=None, model_graph_breakup_param=0.05, one_big_svg_width=500):
    
    if pruning_kwargs_dict is None:
        pruning_kwargs_dict = {}


    # to prevent an error I had, where even the last model would somehow get deleted (which is another error on top of that, because that should never happen)
    if cleanup_k < 1:
        raise ValueError("cleanup_k must be at least 1.")

    
    os.makedirs(main_save_path, exist_ok=True)


    # Research of methods for automatic pruning. 
    # So we could bypass making input_connection_fn and kernel_connection_fn by hand, and make this available for all models out of the box - even when using pretrained ones.
    if False:


        # first method

        # Doesnt work, because i think it does symbolic execution - like, doesnt use real input, it just sees the code and execurtes it in its own interpreter kind of was with made up "proxy" data.
        # And so it cant work with control flow colde (like the if statements i use in the model for padding).
        
        # import torch
        from torch import fx
        from collections import defaultdict


        def trace_layer_usage(model: torch.nn.Module, example_input: torch.Tensor):
            """
            Traces the model using torch.fx and returns a mapping from each Conv2d module
            to the list of nodes (modules) that consume its output.

            Args:
                model: the nn.Module to be traced (with skip or residual connections).
                example_input: a tensor of appropriate shape to run through the model.

            Returns:
                usage_map: dict mapping conv module names to lists of consumer node names.
            """
            # Symbolically trace the model to get a GraphModule
            traced: fx.GraphModule = fx.symbolic_trace(model)
            graph = traced.graph

            # Map from node -> module name for module calls
            node_to_module = {}
            for node in graph.nodes:
                if node.op == 'call_module':
                    node_to_module[node] = node.target  # e.g., 'conv1'

            # Build reverse-lookup: which nodes read from each node
            users_map = defaultdict(list)
            for node in graph.nodes:
                # For each argument of this node, if it's a Node, record usage
                for arg in node.all_input_nodes:
                    users_map[arg].append(node)

            # Now collect usage per Conv2d
            usage_map = {}
            for node in graph.nodes:
                if node.op == 'call_module':
                    mod = dict(model.named_modules())[node.target]
                    if isinstance(mod, torch.nn.Conv2d):
                        # get all users of this conv's output
                        consumers = users_map.get(node, [])
                        usage_map[node.target] = [
                            (user.op + ':' + (user.target if user.op=='call_module' else user.name))
                            for user in consumers
                        ]
            return usage_map


        class SimpleResNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)
                self.conv3 = torch.nn.Conv2d(16, 16, 3, padding=1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                out1 = self.relu(self.conv1(x))
                out2 = self.relu(self.conv2(out1))
                res = out1 + out2
                out3 = self.relu(self.conv3(res))
                return out3

        model = SimpleResNet()
        dummy = torch.randn(1, 3, 224, 224)
        usage = trace_layer_usage(model, dummy)
        for conv_name, consumers in usage.items():
            print(f"{conv_name} output is used by: {consumers}")
        




        # import torch
        from torch.utils.tensorboard import SummaryWriter
        from torchviz import make_dot
        from torch.fx import symbolic_trace


        try:
            model = model_wrapper.model
            dummy_input = model_wrapper.input_example.to(model_wrapper.params.device)

            
            usage = trace_layer_usage(model, dummy_input)
            for conv_name, consumers in usage.items():
                print(f"{conv_name} output is used by: {consumers}")
        except:
            print("FX didnt work.")
            pass
            







        # second method


        activations = {}
        def fw_hook(module, inp, out):
            # activations[f"{module.__class__.__name__}_{id(module)}"] = out.detach()

            # Build a unique key for this module
            key = f"{module.__class__.__name__}_{id(module)}"
            if isinstance(out, tuple):
                # Detach each Tensor in the tuple and store as a tuple
                activations[key] = tuple(
                    elem.detach() if isinstance(elem, torch.Tensor) else elem
                    for elem in out
                )
            else:
                # Single Tensor case
                activations[key] = out.detach()

        for m in model.modules():
            m.register_forward_hook(fw_hook)

        output = model(dummy_input)

        print("Activations:")
        for key, value in activations.items():
            if isinstance(value, tuple):
                print(f"{key}: {[v.shape for v in value]}")
            else:
                print(f"{key}: {value.shape}")





        # third method

        """
        pip install torchviz
    # Additionally, install Graphviz system package:
    # For Ubuntu:
    sudo apt-get install graphviz
    """

        dot = make_dot(output, params=dict(model.named_parameters()))
        dot.format = 'png'
        dot.render('model_graph')



        # Fourth method

        # SummaryWriter().add_graph(model, dummy_input)

        # pip install tensorboard

        # tensorboard --logdir=runs

        
        tb_writer = SummaryWriter(log_dir='runs/your_experiment_name')
        tb_writer.add_graph(model, dummy_input)
        tb_writer.close()
        print("TensorBoard graph written to runs/your_experiment_name")


        
        # import torch.fx
        # torch.fx.wrap('pad_or_resize_to_dims')

        # from torch.fx import symbolic_trace
        # gm = symbolic_trace(model)
        # print("\n===== FX Graph =====\n")
        # for node in gm.graph.nodes:
        #     print(node.format_node())



        # Fifth method
        # import torch
        sm = torch.jit.trace(model, dummy_input)
        for node in sm.graph.nodes():
            for out in node.outputs():
                for use in out.uses():
                    print(node.kind(), '->', use.user.kind())



        # sixth method
        print("\n\n\n sixth method \n\n\n")

        def traverse(fn):
            if fn is None:
                return
            print(fn)
            for next_fn, _ in fn.next_functions:
                traverse(next_fn)

        
        traverse(output.grad_fn)



        # seventh method
        print("\n\n\n seventh method \n\n\n")
        edges = []
        for line in dot.body:
            if '->' in line:
                src, dst = line.split('->')
                edges.append((src.strip(), dst.strip().strip(';')))
        print("Edges:")
        for src, dst in edges:
            print(f"{src} -> {dst}")
        

        # eight method
        print("\n\n\n eight method \n\n\n")
        sm = torch.jit.trace(model, dummy_input)
        for node in sm.graph.nodes():
            for out in node.outputs():
                for use in out.uses():
                    print(node.kind(), '->', use.user.kind())







        # ninth method
        print("\n\n\n ninth method \n\n\n")

        from torch import nn
        from torch.autograd import Function
        from graphviz import Digraph

        def make_dot(output, params=None, show_saved=False):
            """Produce a Graphviz representation of the autograd graph for `output`.

            Args:
                output (Tensor): the output tensor from which to start tracing.
                params (dict[string, Tensor], optional): mapping from parameter names to tensors,
                    so weight nodes can be labeled.
                show_saved (bool): if True, include edges to saved_tensors (for CatBackward, etc.).
            Returns:
                graphviz.Digraph
            """
            if params is None:
                params = {}

            dot = Digraph(format='png')
            seen = set()

            def size_to_str(size):
                return '(' + (', ').join(str(s) for s in size) + ')'

            def add_param_nodes():
                for name, param in params.items():
                    uid = str(id(param))
                    dot.node(uid, label=f"{name}\n{size_to_str(param.size())}", shape='box', style='filled', fillcolor='lightgray')

            def recurse(fn):
                """Recursively traverse from a grad_fn node."""
                if fn is None or fn in seen:
                    return
                seen.add(fn)

                # Create a node for this Function
                uid = str(id(fn))
                label = type(fn).__name__
                dot.node(uid, label)

                # If this Function saved tensors (e.g. CatBackward, SumBackward…), optionally show them
                if show_saved and hasattr(fn, 'saved_tensors'):
                    for idx, t in enumerate(fn.saved_tensors):
                        tid = str(id(t))
                        dot.node(tid, label=f"saved[{idx}]\n{size_to_str(t.size())}", shape='oval', style='dotted')
                        dot.edge(tid, uid, style='dotted')
                
                # Hasn't doen anything.
                # # show atters of Function
                # if show_saved and hasattr(fn, 'attrs'):
                #     for idx, t in enumerate(fn.attrs):
                #         tid = str(id(t))
                #         dot.node(tid, label=f"saved[{idx}]\n{size_to_str(t.size())}", shape='oval', style='dotted')
                #         dot.edge(tid, uid, style='dotted')
                
                

                # Traverse next_functions: edges from dependencies → this node
                for next_fn, _ in fn.next_functions:
                    if next_fn is not None:
                        n_uid = str(id(next_fn))
                        # If next_fn is actually a leaf parameter, link from the param node
                        
                        if type(next_fn).__name__ == 'AccumulateGrad':
                            var = next_fn.variable
                            pname = None
                            for name, p in params.items():
                                if p is var:
                                    pname = name; break
                            if pname is not None:
                                dot.edge(str(id(var)), uid)
                            else:
                                dot.node(str(id(var)), label=f"Leaf\n{size_to_str(var.size())}", shape='oval', style='filled', fillcolor='lightblue')
                                dot.edge(str(id(var)), uid)
                        else:
                            dot.edge(str(id(next_fn)), uid)
                            recurse(next_fn)

            # Add parameter nodes first
            add_param_nodes()

            # Start from output.grad_fn
            recurse(output.grad_fn)

            return dot

        # Simple model with concatenation and sum
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
                self.conv2 = nn.Conv2d(3, 4, 3, padding=1)
            def forward(self, x):
                a = self.conv1(x)
                b = self.conv2(x)
                y = torch.cat([a, b], dim=1)        # CatBackward
                z = a + b                           # AddBackward
                z = torch.cat([z, z], dim=1)        # CatBackward
                return y + z

        toy_model = ToyModel()
        # Warm up with a dummy forward to build the graph
        inp = torch.randn(1, 3, 8, 8, requires_grad=True)
        out = toy_model(inp)

        # Build and render the graph
        dot = make_dot(out, params=dict(toy_model.named_parameters()), show_saved=True)
        dot.render('toy_autograd_graph')


        out = model(dummy_input)
        dot = make_dot(out, params=dict(model.named_parameters()), show_saved=True)
        dot.render('autograd_graph')










        # tenth method
        print("\n\n\n tenth method \n\n\n")


        # import torch
        # import torch.nn as nn
        import collections
        from typing import Dict, List, Tuple, Any, Set

        class OutputTracker:
            """
            Tracks where the output tensor of specified PyTorch modules is used as input
            to other modules within a model during a forward pass.

            Useful for visualizing data flow, especially with skip/residual connections,
            and for dependency analysis before pruning.
            """
            def __init__(self, model: nn.Module, target_modules: Tuple[type, ...] = (nn.Conv2d,)):
                """
                Initializes the tracker.

                Args:
                    model (nn.Module): The PyTorch model to track.
                    target_modules (Tuple[type, ...]): A tuple of nn.Module types
                                                        whose outputs we want to track
                                                        (e.g., (nn.Conv2d, nn.Linear)).
                                                        Also tracks which modules consume these outputs.
                """
                self.model = model
                self.target_modules = target_modules
                self.connections: Dict[str, Set[str]] = collections.defaultdict(set)
                # Use tensor id() as key. Maps tensor ID to the name of the module that produced it.
                self.tensor_origin_map: Dict[int, str] = {}
                self.module_names: Dict[nn.Module, str] = {}
                self._assign_module_names()
                self.hook_handles = []
                self.active = False # Flag to ensure hooks only run during track()

            def _assign_module_names(self):
                """Assigns unique names to all modules in the model."""
                for name, module in self.model.named_modules():
                    # Use the name provided by named_modules, which handles nesting.
                    # Make sure each module instance has a unique reference in the dict.
                    self.module_names[module] = name if name else "model_root" # Assign 'model_root' to the top-level module if unnamed

            def _forward_hook(self, module: nn.Module, inputs: Any, output: Any):
                """
                Forward hook executed *after* module's forward pass.
                Records the origin of the output tensor(s).
                """
                if not self.active: return # Only run when tracking is active

                module_name = self.module_names.get(module, f"Unnamed_{type(module).__name__}_{id(module)}")
                
                # Handle single tensor output and tuple/list outputs
                if isinstance(output, torch.Tensor):
                    self.tensor_origin_map[id(output)] = module_name
                    # print(f"HOOK: Module '{module_name}' produced tensor {id(output)}") # Debug
                elif isinstance(output, (list, tuple)):
                    for i, out_tensor in enumerate(output):
                        if isinstance(out_tensor, torch.Tensor):
                            # Store origin with an index if multiple outputs
                            self.tensor_origin_map[id(out_tensor)] = f"{module_name}_out{i}"
                            # print(f"HOOK: Module '{module_name}_out{i}' produced tensor {id(out_tensor)}") # Debug
                # else: Handle other potential output types if necessary

            def _forward_pre_hook(self, module: nn.Module, inputs: Any):
                """
                Forward pre-hook executed *before* module's forward pass.
                Checks input tensors against the tensor_origin_map to find connections.
                """
                if not self.active: return # Only run when tracking is active

                target_module_name = self.module_names.get(module, f"Unnamed_{type(module).__name__}_{id(module)}")

                # inputs is often a tuple, even with a single input tensor
                input_tensors = []
                if isinstance(inputs, torch.Tensor):
                    input_tensors.append(inputs)
                elif isinstance(inputs, (list, tuple)):
                    for item in inputs:
                        if isinstance(item, torch.Tensor):
                            input_tensors.append(item)
                        # Can recursively check nested lists/tuples if needed
                # else: Handle other potential input structures if necessary

                for input_tensor in input_tensors:
                    tensor_id = id(input_tensor)
                    # print(f"PRE-HOOK: Module '{target_module_name}' received tensor {tensor_id}") # Debug
                    if tensor_id in self.tensor_origin_map:
                        source_module_name = self.tensor_origin_map[tensor_id]
                        # print(f"PRE-HOOK: Tensor {tensor_id} came from '{source_module_name}'. Adding connection: {source_module_name} -> {target_module_name}") # Debug
                        if source_module_name != target_module_name: # Avoid self-loops if logic allows
                            self.connections[source_module_name].add(target_module_name)

            def _attach_hooks(self):
                """Attaches hooks to all relevant modules."""
                self.remove_hooks() # Ensure no duplicate hooks
                for name, module in self.model.named_modules():
                    # Attach hooks to *all* modules to track the flow comprehensively
                    # The filtering happens when interpreting the results later if needed
                    # (or you could filter here based on self.target_modules if you *only*
                    # care about connections originating from or ending at those specific types)
                    handle_pre = module.register_forward_pre_hook(self._forward_pre_hook)
                    handle_post = module.register_forward_hook(self._forward_hook)
                    self.hook_handles.extend([handle_pre, handle_post])

            def remove_hooks(self):
                """Removes all attached hooks."""
                for handle in self.hook_handles:
                    handle.remove()
                self.hook_handles = []

            def reset(self):
                """Clears the tracked connections and tensor map."""
                self.connections.clear()
                self.tensor_origin_map.clear()

            def track(self, *args, **kwargs) -> Dict[str, Set[str]]:
                """
                Performs a forward pass on the model with tracking enabled.

                Args:
                    *args: Positional arguments for the model's forward pass.
                    **kwargs: Keyword arguments for the model's forward pass.

                Returns:
                    Dict[str, Set[str]]: A dictionary where keys are the names of source
                                        modules and values are sets of names of modules
                                        that consume the source module's output tensor(s).
                """
                self.reset()
                self._attach_hooks()
                self.active = True # Enable hooks

                # Perform the forward pass
                try:
                    _ = self.model(*args, **kwargs)
                finally:
                    # Ensure hooks are removed and state reset even if forward pass fails
                    self.active = False # Disable hooks
                    self.remove_hooks()
                    # tensor_origin_map is cleared on next run, keep it for potential inspection
                    # self.tensor_origin_map.clear() # Optionally clear immediately

                # Filter connections to only show those originating from target_modules if desired
                # (Or return all connections as done here)
                # filtered_connections = {
                #    src: targets for src, targets in self.connections.items()
                #    if any(src.startswith(name) for name, mod in self.model.named_modules()
                #           if isinstance(mod, self.target_modules) and self.module_names[mod] == src)
                # }
                # return filtered_connections
                
                # Return all connections found
                # Convert sets to lists for cleaner printing if desired
                return {k: sorted(list(v)) for k, v in self.connections.items()}

            def __enter__(self):
                # Allows using 'with OutputTracker(model) as tracker:'
                self.reset()
                self._attach_hooks()
                self.active = True
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                # Cleans up hooks when exiting 'with' block
                self.active = False
                self.remove_hooks()
                # Optionally clear map: self.tensor_origin_map.clear()


        # --- Example Usage ---

        # Define a simple CNN with skip/residual connections
        class SimpleResNetBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

            def forward(self, x):
                identity = x
                out = self.relu(self.conv1(x))
                out = self.conv2(out)
                out += identity # Residual connection (element-wise add)
                return self.relu(out)

        class SkipCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.initial_conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
                self.relu1 = nn.ReLU()
                self.res_block1 = SimpleResNetBlock(16)
                self.downsample_conv = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
                self.relu2 = nn.ReLU()
                self.res_block2 = SimpleResNetBlock(32)
                # Example of a skip connection via concatenation (different structure)
                self.skip_conv = nn.Conv2d(16, 16, kernel_size=1) # Process skip connection
                self.upsample = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
                # Concatenate upsampled output with skip connection path
                self.final_conv = nn.Conv2d(16 + 16, 1, kernel_size=3, padding=1) # 16 (upsampled) + 16 (skip)
                self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Make output size predictable

            def forward(self, x):
                # Initial processing
                x1 = self.relu1(self.initial_conv(x)) # Output of initial_conv used by relu1

                # First residual block
                x2 = self.res_block1(x1) # Output of res_block1 used by downsample_conv AND skip_conv

                # Downsampling path
                x3 = self.relu2(self.downsample_conv(x2)) # Output used by res_block2
                x4 = self.res_block2(x3) # Output used by upsample

                # Upsampling path
                x5_upsampled = self.upsample(x4) # Output used by final_conv (via concat)

                # Skip connection path
                x6_skip = self.skip_conv(x2) # Output used by final_conv (via concat)

                # Concatenate and final convolution
                # NOTE: The 'torch.cat' operation itself isn't an nn.Module and won't be directly
                #       tracked by this hook-based method. The pre-hook on 'final_conv' will see
                #       the *result* of the concatenation. Its ID won't be in tensor_origin_map.
                #       This tracker identifies module-to-module tensor passing.
                #       It shows x5_upsampled (from upsample) and x6_skip (from skip_conv)
                #       are *available* just before final_conv, implying their use.
                #       A more advanced graph tracer (like torch.fx) might be needed
                #       to explicitly track the 'cat' operation itself.
                x7_cat = torch.cat((x5_upsampled, x6_skip), dim=1)
                out = self.final_conv(x7_cat) # Consumes the result of cat

                out = self.pool(out) # Pool consumes output of final_conv
                return out

        # --- Print the Results ---
        def printer(connections, model):
            print("--- Data Flow Connections ---")
            print("(Source Module Name -> [List of Consumer Module Names])")
            # Focus on Conv2d outputs as requested by the user (filter the results)
            conv_outputs_usage = {k: v for k, v in connections.items() if 'conv' in k.lower()} # Simple name filter

            if not conv_outputs_usage:
                print("\nNo connections found originating from Conv2D layers (or tracking failed).")
                print("\nFull connection map:")
                for source, targets in connections.items():
                    print(f"'{source}' -> {targets}")

            else:
                print("\nUsage of Conv2D Layer Outputs:")
                for source, targets in sorted(conv_outputs_usage.items()):
                    # Find the actual module type for context (optional)
                    try:
                        module = model.get_submodule(source)
                        mod_type = type(module).__name__
                        print(f"'{source}' ({mod_type}) -> {targets}")
                    except AttributeError:
                        # Handle cases like source being 'module_name_out0' for tuple outputs
                        print(f"'{source}' -> {targets}") # Just print names if lookup fails

                print("\n--- Interpretation Notes ---")
                print("- This shows which module's `forward` function received the output tensor")
                print("  from the source module as part of its input tuple.")
                print("- Intermediate operations like `+` or `torch.cat` are not `nn.Module`s and")
                print("  won't appear as explicit sources or targets. However, you can infer their")
                print("  presence by seeing which tensors are consumed by the module *after* the operation.")
                print("  (e.g., 'final_conv' consuming tensors originating from 'upsample' and 'skip_conv')")
        




        # --- Run the Tracking ---
        toy_model = SkipCNN()
        # Create dummy input data (Batch, Channel, Height, Width)
        toy_dummy_input = torch.randn(1, 1, 28, 28)

        # Use the tracker
        tracker = OutputTracker(toy_model, target_modules=(nn.Conv2d,)) # Track outputs of Conv2D

        # Option 1: Direct call
        connections = tracker.track(toy_dummy_input)

        # Option 2: Using 'with' statement (preferred for cleanup)
        # with OutputTracker(model, target_modules=(nn.Conv2d,)) as tracker:
        #     _ = model(dummy_input) # Forward pass happens implicitly via model call
        #     connections = {k: sorted(list(v)) for k, v in tracker.connections.items()}

        printer(connections, toy_model)




        # --- Run the Tracking ---

        # Use the tracker
        tracker = OutputTracker(model, target_modules=(nn.Conv2d,)) # Track outputs of Conv2D

        # Option 1: Direct call
        connections = tracker.track(dummy_input)

        # Option 2: Using 'with' statement (preferred for cleanup)
        # with OutputTracker(model, target_modules=(nn.Conv2d,)) as tracker:
        #     _ = model(dummy_input) # Forward pass happens implicitly via model call
        #     connections = {k: sorted(list(v)) for k, v in tracker.connections.items()}

        printer(connections, model)






        # eleventh method

        print("\n\n\n eleventh method \n\n\n")
        from collections import defaultdict
        consumers = defaultdict(list)

        def hook_fn(mod, inp, out):
            for next_mod in model.modules():
                # simplistic; better match via tensor identity
                if out in next_mod.__inputs__:
                    consumers[mod].append(next_mod)

        for m in model.modules():
            m.register_forward_hook(hook_fn)
        _ = model(dummy_input)

        print("Consumers:")
        for mod, next_mods in consumers.items():
            print(f"{mod.__class__.__name__}_{id(mod)} -> {[f'{next_mod.__class__.__name__}_{id(next_mod)}' for next_mod in next_mods]}")



        # exit the program:
        import sys
        sys.exit(0)




    training_logs = TrainingLogs.load_or_create_training_logs(main_save_path, num_of_epochs_per_training, cleaning_err_key)

    pruning_logs = PruningLogs.load_or_create_pruning_logs(main_save_path)
    
    # We now save pruning every time we prune, so we don't need to clean up the pruning logs.
    # (The confirming flags will still exist, but who cares.)
    # pruning_logs.clean_up_pruning_train_iters()
        


    if training_logs.last_log is not None:
        train_iter = training_logs.last_log["train_iter"]
    else:
        train_iter = 0
    
    initial_train_iter = train_iter







    j_path = osp.join(main_save_path, "initial_train_iters.json")
    j_dict = jh.load(j_path)
    if j_dict is not None:
        j_dict["initial_train_iters"].append(initial_train_iter)
    else:
        j_dict = {"initial_train_iters" : [initial_train_iter]}
    jh.dump(j_path, j_dict)



    # It's nice to get this every time we run our model. This way, we never really have to explicitly run z_get_fw.py.
    log_flops_and_weights(model_wrapper, main_save_path, f"{train_iter}_start")




    num_of_auto_prunings = 0

    while True:



        # Implement the stopping by hand. We need this for debugging.
        
        if train_iter_possible_stop == 0 or ( (train_iter - initial_train_iter) >= train_iter_possible_stop and  (train_iter - initial_train_iter) % train_iter_possible_stop == 0 ):
            
            inp = input(f"""{train_iter_possible_stop} trainings have been done without error stopping.
                        Best k models are kept. (possibly (k+1) models are kept if one of the worse models is the last model we have).
                        Enter bst to do a batch_size_train and re-ask for input.
                        Enter bse to do a batch_size_eval and re-ask for input.
                        Enter sp to do a save_preds from the data_path/save_preds directory, and re-ask for input.
                        Enter da to do a data augmentation showcase and re-ask for input.
                        Enter ts to do a test showcase of the model and re-ask for input.
                        Enter "resource_graph" to trigger resource_graph() and re-ask for input.
                        Enter s to save the model and re-ask for input.
                        Enter g to show the graph of the model and re-ask for input.
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            

            if inp == "fw":
                                
                log_flops_and_weights(model_wrapper, main_save_path, train_iter)

                inp = input(f"""{train_iter_possible_stop} trainings have been done without error stopping.
                            Best k models are kept. (possibly (k+1) models are kept if one of the worse models is the last model we have).
                            Enter bse to do a batch_size_eval and re-ask for input.
                            Enter sp to do a save_preds from the data_path/save_preds directory, and re-ask for input.
                            Enter da to do a data augmentation showcase and re-ask for input.
                            Enter ts to do a test showcase of the model and re-ask for input.
                            Enter "resource_graph" to trigger resource_graph() and re-ask for input.
                            Enter s to save the model and re-ask for input.
                            Enter g to show the graph of the model and re-ask for input.
                            Enter r to trigger show_results() and re-ask for input.
                            Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                            Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                            Press Enter to continue training.
                            Enter any other key to stop.\n""")

            if inp == "bst":
                model_wrapper.training_wrapper.batch_size_train()

                inp = input(f"""{train_iter_possible_stop} trainings have been done without error stopping.
                            Best k models are kept. (possibly (k+1) models are kept if one of the worse models is the last model we have).
                            Enter bse to do a batch_size_eval and re-ask for input.
                            Enter sp to do a save_preds from the data_path/save_preds directory, and re-ask for input.
                            Enter da to do a data augmentation showcase and re-ask for input.
                            Enter ts to do a test showcase of the model and re-ask for input.
                            Enter "resource_graph" to trigger resource_graph() and re-ask for input.
                            Enter s to save the model and re-ask for input.
                            Enter g to show the graph of the model and re-ask for input.
                            Enter r to trigger show_results() and re-ask for input.
                            Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                            Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                            Press Enter to continue training.
                            Enter any other key to stop.\n""")          

            if inp == "bse":
                model_wrapper.training_wrapper.batch_size_eval()

                inp = input(f"""{train_iter_possible_stop} trainings have been done without error stopping.
                            Best k models are kept. (possibly (k+1) models are kept if one of the worse models is the last model we have).
                            Enter sp to do a save_preds from the data_path/save_preds directory, and re-ask for input.
                            Enter da to do a data augmentation showcase and re-ask for input.
                            Enter ts to do a test showcase of the model and re-ask for input.
                            Enter "resource_graph" to trigger resource_graph() and re-ask for input.
                            Enter s to save the model and re-ask for input.
                            Enter g to show the graph of the model and re-ask for input.
                            Enter r to trigger show_results() and re-ask for input.
                            Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                            Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                            Press Enter to continue training.
                            Enter any other key to stop.\n""")
            
            if inp == "sp":
                model_wrapper.training_wrapper.save_preds(path_to_save_to=osp.join(main_save_path, f"{train_iter}_save_preds"))

                inp = input(f"""
                            Enter da to do a data augmentation showcase and re-ask for input.
                            Enter ts to do a test showcase of the model and re-ask for input.
                            Enter "resource_graph" to trigger resource_graph() and re-ask for input.
                            Enter s to save the model and re-ask for input.
                            Enter g to show the graph of the model and re-ask for input.
                            Enter r to trigger show_results() and re-ask for input.
                            Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                            Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                            Press Enter to continue training.
                            Enter any other key to stop.\n""")

            if inp == "da":
                
                inp = ""
                curr_dataset = model_wrapper.training_wrapper.dataloaders_dict["train"].dataset
                target_name = model_wrapper.training_wrapper.params.target
                da_dataloader = DataLoader(curr_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
                img_ix = 0

                save_path = osp.join(main_save_path, "data_aug")
                quick_figs_counter = 0
                while inp == "":
                    
                    ix = 0
                    for data_dict in da_dataloader:
                        
                        # so we actually get to the image we want to see
                        if ix < img_ix:
                            ix += 1
                            continue


                        X = data_dict["images"]
                        y = data_dict[target_name]
                        # img_names = data_dict["img_names"]

                        # curr_img = X[0, :3].squeeze()
                        curr_img = X[0].squeeze()[:3]

                        curr_target = y[0].squeeze()
                        break

                    combined_img = curr_img * (1 - curr_target)   # This will make all vein pixels blacked out.
                    combined_img_2 = curr_img * curr_target       # This will make all non-vein pixels blacked out.
                    save_img(combined_img, save_path, f"da_{quick_figs_counter}_img_vein_blacked_out.png")
                    save_img(combined_img_2, save_path, f"da_{quick_figs_counter}_img_non_vein_blacked_out.png")
                    
                    
                    
                    # fig, _ = show_image([curr_img, curr_target])
                    # save_plt_fig(fig, save_path, f"da_{quick_figs_counter}")

                    curr_img = smart_conversion(curr_img, "ndarray", "uint8")
                    # save_img(curr_img, save_path, f"da_{quick_figs_counter}_img.png")
                    

                    # mask is int64, because torch likes it like that. Lets make it float, because the vals are only 0s and 1s, and so smart conversion in save_img()
                    # will make it 0s and 255s.
                    curr_target = curr_target.to(torch.float32)
                    # save_img(curr_target, save_path, f"da_{quick_figs_counter}_target.png")




                    quick_figs_counter += 1

                    inp = input("""Press Enter to get the next data augmentation. Enter a number to swith to img with that ix in the dataset as the subject of this data augmentation test.
                                Enter anything to stop with the data augmentation testing.\n""")
                    
                    if inp.isdigit():
                        img_ix = int(inp)
                        inp = ""
                    
                    

                inp = input(f"""
                        Enter ts to do a test showcase of the model and re-ask for input.
                        Enter "resource_graph" to trigger resource_graph() and re-ask for input.
                        Enter s to save the model and re-ask for input.
                        Enter g to show the graph of the model and re-ask for input.
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            
            
            if inp == "ts":
                model_wrapper.training_wrapper.test_showcase(path_to_save_to=osp.join(main_save_path, f"{train_iter}_test_showcase"))
                inp = input(f"""
                        Enter "resource_graph" to trigger resource_graph() and re-ask for input.
                        Enter s to save the model and re-ask for input.
                        Enter g to show the graph of the model and re-ask for input.
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            

            
            if inp == "resource_graph":
                res = resource_graph(main_save_path, model_wrapper.save_path)
                
                if res is not None:
                    fig, _, res_dict = res
                    graph_save_path = osp.join(main_save_path, "graphs")
                    os.makedirs(graph_save_path, exist_ok=True)
                    save_plt_fig(fig, graph_save_path, f"{train_iter}_resource_graph")
                    with open(osp.join(graph_save_path, f"{train_iter}_resource_dict.pkl"), "wb") as f:
                        pickle.dump(res_dict, f)

                inp = input(f"""
                        Enter s to save the model and re-ask for input.
                        Enter g to show the graph of the model and re-ask for input.
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            if inp == "s":
                # saving model and reasking for input


                training_logs.delete_all_but_best_k_models(cleanup_k, model_wrapper)
                training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, train_iter, "special_save")
                model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_special_save")
                
                inp = input(f"""
                        Enter g to show the graph of the model and re-ask for input.
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            
            if inp == "g":

                graph_save_path = osp.join(main_save_path, f"graphs")

                # make the plt pickle verion (for viewing thee moddel graph by opening the picke and interacting through plt)
                model_graph_args_dict = {
                    "width": 10,
                }
                list_of_fig_ax_id_tuples = model_wrapper.model_graph(model_graph_args_dict)
                fig = list_of_fig_ax_id_tuples[0][0]
                if shared.PLT_SHOW:
                    fig.show()
                # input("wait")
                save_plt_fig(fig, graph_save_path, f"{train_iter}_model_graph", formats={"pkl"})


                # one big svg
                model_graph_args_dict = {
                    "width": one_big_svg_width,
                    "for_img_dict" : {
                        "min_child_width_limit": 0.0
                    }
                }
                list_of_fig_ax_id_tuples = model_wrapper.model_graph(model_graph_args_dict)
                fig = list_of_fig_ax_id_tuples[0][0]
                save_plt_fig(fig, graph_save_path, f"{train_iter}_model_graph", formats={"svg"})



                # broken up svgs
                # change up min_child_width_limit until you find a nice value for where the graphs split up 
                model_graph_args_dict = {
                    "width": 100,
                    "for_img_dict" : {
                        "min_child_width_limit": model_graph_breakup_param
                    }
                }
                list_of_fig_ax_id_tuples = model_wrapper.model_graph(model_graph_args_dict)
                save_path_for_collection = osp.join(graph_save_path, f"{train_iter}_broken_up_svgs")
                
                count = 1
                while osp.exists(save_path_for_collection):    # to make sure we don't overwrite anything, and we keep things readable
                    save_path_for_collection = osp.join(graph_save_path, f"{train_iter}_broken_up_svgs_{count}")
                    count += 1
                
                for fig, _, curr_id in list_of_fig_ax_id_tuples:
                    save_plt_fig(fig, save_path_for_collection, f"{train_iter}_model_graph_{curr_id}", formats={"svg"})






                
                inp = input("""
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            if inp == "r":
                res = show_results(main_save_path)
                if res is not None:
                    fig, _ = res
                    graph_save_path = osp.join(main_save_path, "graphs")
                    save_plt_fig(fig, graph_save_path, f"{train_iter}_show_results")
                inp = input("""
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            
            try:
                train_iter_possible_stop = int(inp)
                print(f"New trainings before stopping: {train_iter_possible_stop}")
                inp = input("""
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            except ValueError:
                pass


            if inp == "p":
                
                # This will ensure I have the best k models from every pruning phase.

                curr_pickleable_conv_res_calc = model_wrapper.resource_calc.get_copy_for_pickle()

                model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_before_pruning")

                # And this makes even less sense:
                # pruning_logs.log_pruning_train_iter(train_iter, curr_pickleable_conv_res_calc)

                # Makes no sense to also save the model before pruning - it is literally the same model we saved at the end of the previous while.
                # training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, train_iter, "before_pruning")


                curr_pickleable_conv_res_calc, _ = model_wrapper.prune(**pruning_kwargs_dict)

                training_logs.delete_all_but_best_k_models(cleanup_k, model_wrapper)
                pruning_logs.log_pruning_train_iter(train_iter, curr_pickleable_conv_res_calc)
                training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, train_iter, "after_pruning")
                model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_after_pruning")


                inp = input("""
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")

            # if inp == "g":

            #     graph_save_path = osp.join(main_save_path, f"graphs")

            #     # make the plt pickle verion (for viewing thee moddel graph by opening the picke and interacting through plt)
            #     model_graph_args_dict = {
            #         "width": 10,
            #     }
            #     list_of_fig_ax_id_tuples = model_wrapper.model_graph(model_graph_args_dict)
            #     fig = list_of_fig_ax_id_tuples[0][0]
            #     fig.show()
            #     input("wait")
            #     save_plt_fig(fig, graph_save_path, f"{train_iter}_model_graph", formats={"pkl"})


            #     # one big svg
            #     model_graph_args_dict = {
            #         "width": 800,
            #         "for_img_dict" : {
            #             "min_child_width_limit": 0.0
            #         }
            #     }
            #     list_of_fig_ax_id_tuples = model_wrapper.model_graph(model_graph_args_dict)
            #     fig = list_of_fig_ax_id_tuples[0][0]
            #     save_plt_fig(fig, graph_save_path, f"{train_iter}_model_graph", formats={"svg"})



            #     # broken up svgs
            #     # change up min_child_width_limit until you find a nice value for where the graphs split up 
            #     model_graph_args_dict = {
            #         "width": 100,
            #         "for_img_dict" : {
            #             "min_child_width_limit": 0.08
            #         }
            #     }
            #     list_of_fig_ax_id_tuples = model_wrapper.model_graph(model_graph_args_dict)
            #     save_path_for_collection = osp.join(graph_save_path, f"{train_iter}_broken_up_svgs")
            #     for fig, _, curr_id in list_of_fig_ax_id_tuples:
            #         save_plt_fig(fig, save_path_for_collection, f"{train_iter}_model_graph_{curr_id}", formats={"svg"})


            #     inp = input("""
            #             Press Enter to continue training.
            #             Enter any other key to stop.\n""")
            

            if inp != "":
                break



        # The pruning mechanism

        if pruning_phase and val_stop_fn(training_logs, pruning_logs, train_iter, initial_train_iter):
            
            curr_pickleable_conv_res_calc = model_wrapper.resource_calc.get_copy_for_pickle()



            # This is nice and all, but I always just use after pruning ones. It's just easier, and this here wouldnt save enough training iterations. Sorry.
            # # This will ensure I have the best k models from every pruning phase.
            # model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_before_pruning")

            # And this makes even less sense:
            # pruning_logs.log_pruning_train_iter(train_iter, curr_pickleable_conv_res_calc)

            # Makes no sense to also save the model before pruning - it is literally the same model we saved at the end of the previous while.
            # training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, train_iter, "before_pruning")

            
            curr_pickleable_conv_res_calc, are_there_more_to_prune_in_the_future = model_wrapper.prune(**pruning_kwargs_dict)

            num_of_auto_prunings += 1




            # This will ensure I have the best k models from every pruning phase.
            training_logs.delete_all_but_best_k_models(cleanup_k, model_wrapper)
            pruning_logs.log_pruning_train_iter(train_iter, curr_pickleable_conv_res_calc)
            training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, train_iter, "after_pruning")

            info_dict = model_wrapper.get_resource_info(pruning_kwargs_dict["resource_name"])
            curr_resource_percentage = info_dict["percentage"]
            if viscinity_save_params is None or viscinity_save_check(viscinity_save_params, curr_resource_percentage):
                # training_logs.delete_all_but_best_k_models(cleanup_k, model_wrapper)
                model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_after_pruning")



            if not are_there_more_to_prune_in_the_future:
                print("There are no more kernels that could be pruned in the future.")
                break



        if train_iter >= max_total_training_iters:
            print(f"Max total training iterations reached: {max_total_training_iters}. Train_iter: {train_iter}")
            break
        
        if (train_iter - initial_train_iter) >= max_training_iters:
            print(f"Max training iterations reached: {max_training_iters}. Train_iter: {train_iter}, Initial_train_iter: {initial_train_iter}")
            break

        if num_of_auto_prunings >= max_auto_prunings:
            print(f"Max auto prunings reached: {max_auto_prunings}. num_of_auto_prunings: {num_of_auto_prunings}")
            break
    









        
        returnings = model_wrapper.train(num_of_epochs_per_training)
        train_error = returnings[0]["metrics_dict"]

        # print(f"Hooks: {model_wrapper.tree_ix_2_hook_handle}")

        val_error = model_wrapper.validation()
        test_error = model_wrapper.test()


        train_iter += 1 # this reflects how many trainings we have done

        print(f"We have finished training iteration {train_iter}")



        training_logs.delete_all_but_best_k_models(cleanup_k, model_wrapper)
        training_logs, pruning_logs = perform_save(model_wrapper, training_logs, pruning_logs, train_iter, "", val_error, test_error, train_error)
        # training_logs.delete_all_but_best_k_models(cleanup_k, model_wrapper)






    # After the while loop is broken out of:
    log_flops_and_weights(model_wrapper, main_save_path, f"{train_iter}_end")
    model_wrapper.create_safety_copy_of_existing_models(f"{train_iter}_ending_save")
        





# calculate_resources(self, input_example):

```
	y = self.wrapper_model.model.forward(input_example)
        restore_forward(self.wrapper_model.model)
        print(self.module_tree_ixs_2_name)
        print(self.module_tree_ixs_2_flops_dict)
        recursively_populate_resources()
        print(self.module_tree_ixs_2_flops_dict)
        print(self.cur_flops)
```
The output is broken into some newlines for readability:
```
{None: 'unet', (None, 0): 'doubleconv', 
((None, 0), 0): 'sequential', 
(((None, 0), 0), 0): 'conv2d', 
(((None, 0), 0), 1): 'batchnorm2d', 
(((None, 0), 0), 2): 'relu', 
(((None, 0), 0), 3): 'conv2d', 
(((None, 0), 0), 4): 'batchnorm2d', 
(((None, 0), 0), 5): 'relu', 
(None, 1): 'down', 
((None, 1), 0): 'sequential', 
(((None, 1), 0), 0): 'maxpool2d', (((None, 1), 0), 1): 'doubleconv', ((((None, 1), 0), 1), 0): 'sequential', (((((None, 1), 0), 1), 0), 0): 'conv2d', (((((None, 1), 0), 1), 0), 1): 'batchnorm2d', (((((None, 1), 0), 1), 0), 2): 'relu', (((((None, 1), 0), 1), 0), 3): 'conv2d', (((((None, 1), 0), 1), 0), 4): 'batchnorm2d', (((((None, 1), 0), 1), 0), 5): 'relu', (None, 2): 'down', ((None, 2), 0): 'sequential', (((None, 2), 0), 0): 'maxpool2d', (((None, 2), 0), 1): 'doubleconv', ((((None, 2), 0), 1), 0): 'sequential', (((((None, 2), 0), 1), 0), 0): 'conv2d', (((((None, 2), 0), 1), 0), 1): 'batchnorm2d', (((((None, 2), 0), 1), 0), 2): 'relu', (((((None, 2), 0), 1), 0), 3): 'conv2d', (((((None, 2), 0), 1), 0), 4): 'batchnorm2d', (((((None, 2), 0), 1), 0), 5): 'relu', (None, 3): 'down', ((None, 3), 0): 'sequential', (((None, 3), 0), 0): 'maxpool2d', (((None, 3), 0), 1): 'doubleconv', ((((None, 3), 0), 1), 0): 'sequential', (((((None, 3), 0), 1), 0), 0): 'conv2d', (((((None, 3), 0), 1), 0), 1): 'batchnorm2d', (((((None, 3), 0), 1), 0), 2): 'relu', (((((None, 3), 0), 1), 0), 3): 'conv2d', (((((None, 3), 0), 1), 0), 4): 'batchnorm2d', (((((None, 3), 0), 1), 0), 5): 'relu', (None, 4): 'down', ((None, 4), 0): 'sequential', (((None, 4), 0), 0): 'maxpool2d', (((None, 4), 0), 1): 'doubleconv', ((((None, 4), 0), 1), 0): 'sequential', (((((None, 4), 0), 1), 0), 0): 'conv2d', (((((None, 4), 0), 1), 0), 1): 'batchnorm2d', (((((None, 4), 0), 1), 0), 2): 'relu', (((((None, 4), 0), 1), 0), 3): 'conv2d', (((((None, 4), 0), 1), 0), 4): 'batchnorm2d', (((((None, 4), 0), 1), 0), 5): 'relu', (None, 5): 'up', ((None, 5), 0): 'upsample', ((None, 5), 1): 'doubleconv', (((None, 5), 1), 0): 'sequential', ((((None, 5), 1), 0), 0): 'conv2d', ((((None, 5), 1), 0), 1): 'batchnorm2d', ((((None, 5), 1), 0), 2): 'relu', ((((None, 5), 1), 0), 3): 'conv2d', ((((None, 5), 1), 0), 4): 'batchnorm2d', ((((None, 5), 1), 0), 5): 'relu', (None, 6): 'up', ((None, 6), 0): 'upsample', ((None, 6), 1): 'doubleconv', (((None, 6), 1), 0): 'sequential', ((((None, 6), 1), 0), 0): 'conv2d', ((((None, 6), 1), 0), 1): 'batchnorm2d', ((((None, 6), 1), 0), 2): 'relu', ((((None, 6), 1), 0), 3): 'conv2d', ((((None, 6), 1), 0), 4): 'batchnorm2d', ((((None, 6), 1), 0), 5): 'relu', (None, 7): 'up', ((None, 7), 0): 'upsample', ((None, 7), 1): 'doubleconv', (((None, 7), 1), 0): 'sequential', ((((None, 7), 1), 0), 0): 'conv2d', ((((None, 7), 1), 0), 1): 'batchnorm2d', ((((None, 7), 1), 0), 2): 'relu', ((((None, 7), 1), 0), 3): 'conv2d', ((((None, 7), 1), 0), 4): 'batchnorm2d', ((((None, 7), 1), 0), 5): 'relu', (None, 8): 'up', ((None, 8), 0): 'upsample', ((None, 8), 1): 'doubleconv', (((None, 8), 1), 0): 'sequential', ((((None, 8), 1), 0), 0): 'conv2d', ((((None, 8), 1), 0), 1): 'batchnorm2d', ((((None, 8), 1), 0), 2): 'relu', ((((None, 8), 1), 0), 3): 'conv2d', ((((None, 8), 1), 0), 4): 'batchnorm2d', ((((None, 8), 1), 0), 5): 'relu', (None, 9): 'outconv', ((None, 9), 0): 'conv2d'}


{None: 0, (None, 0): 0, 
((None, 0), 0): 0, 
(((None, 0), 0), 0): 9437184, 
(((None, 0), 0), 1): 0, 
(((None, 0), 0), 2): 0, 
(((None, 0), 0), 3): 603979776, 
(((None, 0), 0), 4): 0, 
(((None, 0), 0), 5): 0, (None, 1): 0, 
((None, 1), 0): 0, (((None, 1), 0), 0): 0, 
(((None, 1), 0), 1): 0, 
((((None, 1), 0), 1), 0): 0, 
(((((None, 1), 0), 1), 0), 0): 301989888, 
(((((None, 1), 0), 1), 0), 1): 0, 
(((((None, 1), 0), 1), 0), 2): 0, (((((None, 1), 0), 1), 0), 3): 603979776, (((((None, 1), 0), 1), 0), 4): 0, (((((None, 1), 0), 1), 0), 5): 0, (None, 2): 0, ((None, 2), 0): 0, (((None, 2), 0), 0): 0, (((None, 2), 0), 1): 0, ((((None, 2), 0), 1), 0): 0, (((((None, 2), 0), 1), 0), 0): 301989888, (((((None, 2), 0), 1), 0), 1): 0, (((((None, 2), 0), 1), 0), 2): 0, (((((None, 2), 0), 1), 0), 3): 603979776, (((((None, 2), 0), 1), 0), 4): 0, (((((None, 2), 0), 1), 0), 5): 0, (None, 3): 0, ((None, 3), 0): 0, (((None, 3), 0), 0): 0, (((None, 3), 0), 1): 0, ((((None, 3), 0), 1), 0): 0, (((((None, 3), 0), 1), 0), 0): 301989888, (((((None, 3), 0), 1), 0), 1): 0, (((((None, 3), 0), 1), 0), 2): 0, (((((None, 3), 0), 1), 0), 3): 603979776, (((((None, 3), 0), 1), 0), 4): 0, (((((None, 3), 0), 1), 0), 5): 0, (None, 4): 0, ((None, 4), 0): 0, (((None, 4), 0), 0): 0, (((None, 4), 0), 1): 0, ((((None, 4), 0), 1), 0): 0, (((((None, 4), 0), 1), 0), 0): 150994944, (((((None, 4), 0), 1), 0), 1): 0, (((((None, 4), 0), 1), 0), 2): 0, (((((None, 4), 0), 1), 0), 3): 150994944, (((((None, 4), 0), 1), 0), 4): 0, (((((None, 4), 0), 1), 0), 5): 0, (None, 5): 0, ((None, 5), 0): 0, ((None, 5), 1): 0, (((None, 5), 1), 0): 0, ((((None, 5), 1), 0), 0): 1207959552, ((((None, 5), 1), 0), 1): 0, ((((None, 5), 1), 0), 2): 0, ((((None, 5), 1), 0), 3): 301989888, ((((None, 5), 1), 0), 4): 0, ((((None, 5), 1), 0), 5): 0, (None, 6): 0, ((None, 6), 0): 0, ((None, 6), 1): 0, (((None, 6), 1), 0): 0, ((((None, 6), 1), 0), 0): 1207959552, ((((None, 6), 1), 0), 1): 0, ((((None, 6), 1), 0), 2): 0, ((((None, 6), 1), 0), 3): 301989888, ((((None, 6), 1), 0), 4): 0, ((((None, 6), 1), 0), 5): 0, (None, 7): 0, ((None, 7), 0): 0, ((None, 7), 1): 0, (((None, 7), 1), 0): 0, ((((None, 7), 1), 0), 0): 1207959552, ((((None, 7), 1), 0), 1): 0, ((((None, 7), 1), 0), 2): 0, ((((None, 7), 1), 0), 3): 301989888, ((((None, 7), 1), 0), 4): 0, ((((None, 7), 1), 0), 5): 0, (None, 8): 0, ((None, 8), 0): 0, ((None, 8), 1): 0, (((None, 8), 1), 0): 0, ((((None, 8), 1), 0), 0): 1207959552, ((((None, 8), 1), 0), 1): 0, ((((None, 8), 1), 0), 2): 0, ((((None, 8), 1), 0), 3): 603979776, ((((None, 8), 1), 0), 4): 0, ((((None, 8), 1), 0), 5): 0, (None, 9): 0, ((None, 9), 0): 2097152}


{None: 9977200640, 
(None, 0): 613416960, 
((None, 0), 0): 613416960, 
(((None, 0), 0), 0): 9437184, 
(((None, 0), 0), 1): 0, 
(((None, 0), 0), 2): 0, 
(((None, 0), 0), 3): 603979776, 
(((None, 0), 0), 4): 0, 
(((None, 0), 0), 5): 0, 
(None, 1): 905969664, 
((None, 1), 0): 905969664, 
(((None, 1), 0), 0): 0, 
(((None, 1), 0), 1): 905969664, 
((((None, 1), 0), 1), 0): 905969664, 
(((((None, 1), 0), 1), 0), 0): 301989888, 
(((((None, 1), 0), 1), 0), 1): 0, 
(((((None, 1), 0), 1), 0), 2): 0, 
(((((None, 1), 0), 1), 0), 3): 603979776, 
(((((None, 1), 0), 1), 0), 4): 0, 
(((((None, 1), 0), 1), 0), 5): 0, 
(None, 2): 905969664, 
((None, 2), 0): 905969664, 
(((None, 2), 0), 0): 0, (((None, 2), 0), 1): 905969664, ((((None, 2), 0), 1), 0): 905969664, (((((None, 2), 0), 1), 0), 0): 301989888, (((((None, 2), 0), 1), 0), 1): 0, (((((None, 2), 0), 1), 0), 2): 0, (((((None, 2), 0), 1), 0), 3): 603979776, (((((None, 2), 0), 1), 0), 4): 0, (((((None, 2), 0), 1), 0), 5): 0, (None, 3): 905969664, ((None, 3), 0): 905969664, (((None, 3), 0), 0): 0, (((None, 3), 0), 1): 905969664, ((((None, 3), 0), 1), 0): 905969664, (((((None, 3), 0), 1), 0), 0): 301989888, (((((None, 3), 0), 1), 0), 1): 0, (((((None, 3), 0), 1), 0), 2): 0, (((((None, 3), 0), 1), 0), 3): 603979776, (((((None, 3), 0), 1), 0), 4): 0, (((((None, 3), 0), 1), 0), 5): 0, (None, 4): 301989888, ((None, 4), 0): 301989888, (((None, 4), 0), 0): 0, (((None, 4), 0), 1): 301989888, ((((None, 4), 0), 1), 0): 301989888, (((((None, 4), 0), 1), 0), 0): 150994944, (((((None, 4), 0), 1), 0), 1): 0, (((((None, 4), 0), 1), 0), 2): 0, (((((None, 4), 0), 1), 0), 3): 150994944, (((((None, 4), 0), 1), 0), 4): 0, (((((None, 4), 0), 1), 0), 5): 0, (None, 5): 1509949440, ((None, 5), 0): 0, ((None, 5), 1): 1509949440, (((None, 5), 1), 0): 1509949440, ((((None, 5), 1), 0), 0): 1207959552, ((((None, 5), 1), 0), 1): 0, ((((None, 5), 1), 0), 2): 0, ((((None, 5), 1), 0), 3): 301989888, ((((None, 5), 1), 0), 4): 0, ((((None, 5), 1), 0), 5): 0, (None, 6): 1509949440, ((None, 6), 0): 0, ((None, 6), 1): 1509949440, (((None, 6), 1), 0): 1509949440, ((((None, 6), 1), 0), 0): 1207959552, ((((None, 6), 1), 0), 1): 0, ((((None, 6), 1), 0), 2): 0, ((((None, 6), 1), 0), 3): 301989888, ((((None, 6), 1), 0), 4): 0, ((((None, 6), 1), 0), 5): 0, (None, 7): 1509949440, ((None, 7), 0): 0, ((None, 7), 1): 1509949440, (((None, 7), 1), 0): 1509949440, ((((None, 7), 1), 0), 0): 1207959552, ((((None, 7), 1), 0), 1): 0, ((((None, 7), 1), 0), 2): 0, ((((None, 7), 1), 0), 3): 301989888, ((((None, 7), 1), 0), 4): 0, ((((None, 7), 1), 0), 5): 0, (None, 8): 1811939328, ((None, 8), 0): 0, ((None, 8), 1): 1811939328, (((None, 8), 1), 0): 1811939328, ((((None, 8), 1), 0), 0): 1207959552, ((((None, 8), 1), 0), 1): 0, ((((None, 8), 1), 0), 2): 0, ((((None, 8), 1), 0), 3): 603979776, ((((None, 8), 1), 0), 4): 0, ((((None, 8), 1), 0), 5): 0, (None, 9): 2097152, ((None, 9), 0): 2097152}


9977200640
```


## restore_forward(model):
This is an inner function.

```
            
            for child in model.children():
                # leaf node
                if self._is_leaf(child) and hasattr(child, 'old_forward'):
                    child.forward = child.old_forward
                    child.old_forward = None
                else:
                    restore_forward(child)

            print(10*"\n" + "Children:")
            for child in model.children():
                print(child)
```

```
Children:
Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)










Children:
Sequential(
  (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): ReLU(inplace=True)
)










Children:
Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)










Children:
Sequential(
  (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): ReLU(inplace=True)
)










Children:
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
DoubleConv(
  (double_conv): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
)










Children:
Sequential(
  (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (1): DoubleConv(
    (double_conv): Sequential(
      (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
  )
)










Children:
Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)
Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)










Children:
Sequential(
  (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): ReLU(inplace=True)
)










Children:
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
DoubleConv(
  (double_conv): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
)










Children:
Sequential(
  (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (1): DoubleConv(
    (double_conv): Sequential(
      (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
  )
)










Children:
Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)










Children:
Sequential(
  (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): ReLU(inplace=True)
)










Children:
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
DoubleConv(
  (double_conv): Sequential(
    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
)










Children:
Sequential(
  (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (1): DoubleConv(
    (double_conv): Sequential(
      (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
  )
)










Children:
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)










Children:
Sequential(
  (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): ReLU(inplace=True)
)










Children:
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
DoubleConv(
  (double_conv): Sequential(
    (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
)










Children:
Sequential(
  (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (1): DoubleConv(
    (double_conv): Sequential(
      (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
  )
)










Children:
Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)
Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)










Children:
Sequential(
  (0): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): ReLU(inplace=True)
)










Children:
Upsample(scale_factor=2.0, mode='bilinear')
DoubleConv(
  (double_conv): Sequential(
    (0): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
)










Children:
Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)
Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)










Children:
Sequential(
  (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): ReLU(inplace=True)
)










Children:
Upsample(scale_factor=2.0, mode='bilinear')
DoubleConv(
  (double_conv): Sequential(
    (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
)










Children:
Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)
Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)










Children:
Sequential(
  (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): ReLU(inplace=True)
)










Children:
Upsample(scale_factor=2.0, mode='bilinear')
DoubleConv(
  (double_conv): Sequential(
    (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
)










Children:
Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
ReLU(inplace=True)










Children:
Sequential(
  (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): ReLU(inplace=True)
)










Children:
Upsample(scale_factor=2.0, mode='bilinear')
DoubleConv(
  (double_conv): Sequential(
    (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
)










Children:
Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1))










Children:
DoubleConv(
  (double_conv): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
)
Down(
  (maxpool_conv): Sequential(
    (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (1): DoubleConv(
      (double_conv): Sequential(
        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
      )
    )
  )
)
Down(
  (maxpool_conv): Sequential(
    (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (1): DoubleConv(
      (double_conv): Sequential(
        (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
      )
    )
  )
)
Down(
  (maxpool_conv): Sequential(
    (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (1): DoubleConv(
      (double_conv): Sequential(
        (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
      )
    )
  )
)
Down(
  (maxpool_conv): Sequential(
    (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (1): DoubleConv(
      (double_conv): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
      )
    )
  )
)
Up(
  (up): Upsample(scale_factor=2.0, mode='bilinear')
  (conv): DoubleConv(
    (double_conv): Sequential(
      (0): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
  )
)
Up(
  (up): Upsample(scale_factor=2.0, mode='bilinear')
  (conv): DoubleConv(
    (double_conv): Sequential(
      (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
  )
)
Up(
  (up): Upsample(scale_factor=2.0, mode='bilinear')
  (conv): DoubleConv(
    (double_conv): Sequential(
      (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
  )
)
Up(
  (up): Upsample(scale_factor=2.0, mode='bilinear')
  (conv): DoubleConv(
    (double_conv): Sequential(
      (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
  )
)
OutConv(
  (conv): Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1))
)

```

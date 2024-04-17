pip install -r requirements.txt
-> nekateri ne grejo skoz (sem jih pobrisal) zato jih je treba na roke PIL -> Pillow in tako naprej


conda install pytorch torchvision cudatoolkit=9.2 -c pytorch -c defaults -c numba/label/dev


-------------------------------------- v env:
pip3 install https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-win_amd64.whl
pip install Pillow
pip install torchvision
pip install opencv-python
pip install sklearn
pip install matplotlib
pip install torchsummary

--------------------------------------------

RUN_BEST_MODEL1:
vzel sem best_model od njih, na naši bazi (train=1283, validation=446, test=245) sem pognal train za 60 epochov.
Epoch:60, Train mIoU: 0.690960423952861
Epoch:60, Valid Loss: 0.543 mIoU: 0.7161237460429578 Complexity: 248900 total: 0.8580618730214788



RUN BEST MODEL  SECOND
enako kot prejšnji, spremenil sem računanje IoU, epochs=100, začetek na best_model_original


RUN:
python train.py --dataset eyes --useGPU true --bs 4 --expname RUN_BEST_MODEL --load best_model.pkl  --resume --epochs 100









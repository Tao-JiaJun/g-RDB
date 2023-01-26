# g-RDB
Group Residual Dense Block for Key-Point Detector with One-level Feature

For users in China, I suggest you use Tianyi Cloud.
https://cloud.189.cn/t/mYJ3q2qABvU3 [eme5]

Baidu Cloud
https://pan.baidu.com/s/1H0QohUu53sNWdOOZIGKFRw [GRDB]


Google Driver
https://drive.google.com/drive/folders/15NYnNGLl87c_QN0xdqiuIHq8sOUywWYZ?usp=sharing

## requirements
```python
jpeg4py==0.1.4
matplotlib==3.4.2
numpy==1.21.2
opencv_python==4.5.2.52
pycocotools==2.0.6
thop==0.0.5.post2203101510
torch==1.8.1+cu102
torchvision==0.9.1+cu102
tqdm==4.61.1
```

## How to use

```bash
git clone 
cd ./g-RDB
conda activate $your_env

# For evaluate
python eval_voc.py

# Test VOC image
python test_img.py
```
## Note
1. The original code uses the jpeg4py module, which requires the libjpeg-turbo library. If you don't have this library or don't want to compile it, you can use the following code to replace the "load_img_targets" function in "datasets/voc.py".

- jpeg4py

```python
    def load_img_targets(self, img_id):
        target = self.json_data[img_id[1]]
        # img = jpeg.JPEG(self._imgpath % img_id).decode()
        height, width, _ = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
```
- opencv

```python
 def load_img_targets(self, img_id):
    target = self.json_data[img_id[1]]
    img = cv2.imread(self._imgpath % img_id)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    if self.target_transform is not None:
        target = self.target_transform(target, width, height)
```
# AI期末專案-台灣車牌辨識系統  
本專案之目的為針對台灣車牌進行辨識  

**附件檔案**  
* "2.mp4":本專案用來進行測試之短影片  
* "AI_NumberPlate.ipynb":以Colab開發之完整程式碼  
* "coco.yaml":針對本專案所修改及使用之coco檔  
* "predefined_classes.txt":本專案所使用之labelimg data檔  

### 程式說明
'''py
#Training
%cd drive/MyDrive/yolov7/
!python train.py --workers 1 --device 0 --batch-size 16 --epochs 100 --img 640 640 --hyp data/hyp.scratch.custom.yaml --name yolov7-custom --weights 
'''

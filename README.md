## 提取mel谱 
批量提取mel谱请参考get_single_mel实现mel_extractor.py中的get_file_mel函数，示例为提取test.wav的mel谱，并保存到test.npy  
## vocoder 
generate.py为使用higiGAN将test.npy还原为test_synthesize.wav
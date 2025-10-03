# EUReg: End-to-End Framework for Efficient 2D-3D Ultrasound Registration (MICCAI2025) 

By Haiqiao Wang & Yi Wang

Paper link: [[link]](https://link.springer.com/chapter/10.1007/978-3-032-04937-7_17)

<img width="685" height="179" alt="image" src="https://github.com/user-attachments/assets/39d15db4-7836-4b7d-85fa-0dea1c592c3f" />

## Correspondence

| Dirname  | Experiment Settings| Data link |
| :----------: | :-----------:| :-----------:|
| CAMUS_same   | i   | [[link]](https://drive.usercontent.google.com/download?id=1Jmyl7KsPmwh16VJKt4v4Et_c4-RkaAIJ&export=download&authuser=0)|
| CAMUS_diff   | iii (EUReg10), iv (EUReg20)   | [[link]](https://drive.usercontent.google.com/download?id=1hDqsefF_KSxWwIAPFyMTmAPK3ACvFuVO&export=download&authuser=0)|
| proreg_same   | ii   | [[link]](https://drive.usercontent.google.com/download?id=1optAZ9a9bFl83K9yhmqDFqp1TAUMH9xo&export=download&authuser=0)  |
| proreg_diff   | v (EUReg10), vi (EUReg20)  | [[link]](https://drive.usercontent.google.com/download?id=1optAZ9a9bFl83K9yhmqDFqp1TAUMH9xo&export=download&authuser=0)(same as previous)|

<img width="1559" height="306" alt="image" src="https://github.com/user-attachments/assets/e4df2049-a4b8-4dec-a5fa-ac0cdcbc7909" />

## Instruction

After downloading the dataset, change the 'train_dir' and 'val_dir' in train_xx.py and infer_xx.py, and then run them.

For CAMUS_same/CUReg, we directly used the results from [CUReg](https://github.com/LLEIHIT/CU-Reg).

## Citation
If you find the code useful, please cite our paper.
```
@inproceedings{wang2025eureg,
  title={EUReg: End-to-End Framework for Efficient 2D-3D Ultrasound Registration},
  author={Wang, Haiqiao and Wang, Yi},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={175--185},
  year={2025},
  organization={Springer}
}
```

Our code and data are partially adopted from [CUReg](https://github.com/LLEIHIT/CU-Reg), [FVRNet](https://github.com/DIAL-RPI/FVR-Net), [CAMUS](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8), and [ProReg](https://muregpro.github.io/data.html). Thanks for their work.

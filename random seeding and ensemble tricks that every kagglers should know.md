### random seeding and ensemble tricks that every kagglers should know

https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/214959

by蛙神

![img](https://www.microsoft.com/en-us/research/uploads/prod/2021/01/Ensemble_Figre2_updated-1024x532.jpg)

![img](https://www.microsoft.com/en-us/research/uploads/prod/2021/01/Figure1_esemble-blog-1024x388.jpg)

another trick that i common used is to train a resnet to match results of efficient-net (i.e. different architecture).
actually, I used to explain this phenomenon as bias shifting (my own unproven theory)

- additional minimizing anything else than the data loss is good (regularisation). best parameters for best test loss is close to that of best train loss but these two sets of parameters are not equal (unless perfect distribution).
- if there is domain shift, then there is also parameters shift

IMHO, the difference between green and orange is

green:

- You have 10 models with different initializations (i.e. different seeds) at the same time. It means that you define 10 models in a script.
- train these models together by using loss((F1+F2+…+F10)/10, gt).
- output: (F1+F2+…+F10)/10

orange (I think this is a normal ensemble method):

- You have 10 models with different initializations (i.e. different seeds) separately. It means that you have a model in a script and execute it 10 times.
- train each model by using loss(Fi, gt).
- output (by taking the average): (F1+F2+…+F10)/10
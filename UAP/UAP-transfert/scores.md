# Scores of the UAP on the special dataset (celeb1..11)

|  Original \ DeepFake probability  | no modification| UAP-PGD applied (`uap-pgd_delta=0.04_eps=0.05_max-iter=60.npy`) | UAP-DeepFool applied (`uap-df_delta=0.03_overshoot=0.05.npy`) |
|---------|----------------|-----------------------------------------------------|---------------------------------------------------|
| ![Image0](./imgs/celeb0.jpg)  | 99%            | 1%                                                  | 99%                                               |
| ![Image1](./imgs/celeb1.jpg)  | 99%            | 52%                                                 | 77%                                               |
| ![Image2](./imgs/celeb2.jpg)  | 99%            | 2%                                                  | 99%                                               |
| ![Image3](./imgs/celeb3.jpg)  | 85%            | 1%                                                  | 99%                                               |
| ![image4](./imgs/celeb4.jpg)  | 99%            | 1%                                                  | 99%                                               |
| ![Image5](./imgs/celeb5.jpg)  | 99%            | 98%                                                 | 99%                                               |
| ![Image6](./imgs/celeb6.jpg)  | 94%            | 1%                                                  | 99%                                               |
| ![Image7](./imgs/celeb7.jpg)  | 94%            | 2%                                                  | 99%                                               |
| ![Image8](./imgs/celeb8.jpg)  | 99%            | 2%                                                  | 99%                                               |
| ![Image9](./imgs/celeb9.jpg)  | 99%            | 1%                                                  | 99%                                               |
| ![Image10](./imgs/celeb10.jpg) | 99%            | 77%                                                 | 99%                                               |
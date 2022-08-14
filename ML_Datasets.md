# Machine Learning Datasets
I wanted to be sure that my data was ethically sourced and collected. We informed the people involved to be aware of the extent to which their image, likeness, and representation could be use in machine learning. The data would go no further than me - not to be sold or available for others to use.

Each person stood in front of a camera for a length of time between 10 and 30 min. They were prompted to tell a story of their happiest moments.

This source video was down-sampled into 1 frame per second (fps) and 10 fps sequences. The images were further resized from 16:9 aspect ratio to 1:1 (square) with a power of two width and height (1024x1024). The 1 fps sequence was used to quickly pre-train the NN and set a baseline. Then the 10 fps image data set was used for adding further detail into the NN.

With the training process, each person became their own unique data set rather than to sample everyone together and avoid the flicker face data set[^1] look. Moreover, rather than being overtly selective, my process was to be as inclusive as possible by not dropping any images from the image sequences. I wanted to include the outliers and difficult images, our good sides and bad sides, and each person's individuality/behaviors. This most likely created difficulty in training quickly or becoming _algorithmically accurate_, but I felt like all of the bits were important to include, not just a gestalt.

Both augmentation and mirroring became issues during the training phase. The auto-augmentation caused training scores to bounce, meaning that I would occassionally loose an NN output into a blurry morass. I had to completely prevent augmentation from happening to reach my results. Mirroring created another issue. While mirroring did assist with creating a larger data set to train from, when testing latent-space walks, my people would subtly flip or have eye bulges during transitions. Again, here, mirroring was turned off so that our people data would always look like themselves rather than some kind of self-chimera.

Samples of the datasets during training:
- [Set 1](./set1.md)
- [Set 2](./set2.md)
- [Set 3](./set3.md)
- [Set 4](./set4.md)
- [Set 5](./set5.md)

FID is a way of assessing how close a stylegan generated image matches real image. Lower numbers are better. FID scores for the Flicker Face High Quality (FFHQ) is typically between 4 and 2.

Final  Fréchet Inception Distance (FiD) scores:
Set 1   9.
Set 2   20.
Set 3   114.385   not used in final animation
Set 4   7.
Set 5   30.


[^1]: [https://github.com/NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset)

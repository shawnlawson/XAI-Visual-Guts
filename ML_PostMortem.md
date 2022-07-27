# Looking Back

Going into this project I had a lot of concerns about machine learning (ML) with regards to sourcing the data used in training, curating that data for training, understanding what/how ML is doing/computing, and results were affecting people. Now at the end, those concerns are re-affirmed.

## Sourcing Data

Despite there existing a large number of open source datasets, [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets), I felt it was best to capture my own data set for this project. Even if the data sets are open source, how do I know who's data it really is? Who owns it? Did they give permission? Do they even know? For example, look at some pre-built neural nets (NN), [https://github.com/justinpinkney/awesome-pretrained-stylegan2](https://github.com/justinpinkney/awesome-pretrained-stylegan2). Some of these have copyright information and some don't. Some are created from other artist's work without their knowledge. I can now create an near infinite number of paintings in the likeness of another artist's style. Is this acceptable? 

I wanted to be sure that my data was ethically sourced and collected. I wanted those people involved as data to be aware of the extent to which their image, likeness, representation could be use in machine learning. The data would go no further than me - not to be sold or available for others to use. 


Data Feminism

## Curating Data


training, which images from a set, how many, easy to drop those that don't work (outliers, non-conformers)

technical - mirroring, likeness, loss of humanness from movement. Holes/gaps in latent space. Some quick, some not, 



## Understanding Data

I decided to train a separate NN per person rather than train everyone together. In this way, I would be able to more effectively make comparisons. To my surprise I found shocking differences between NNs during the training process. I had expected each data set's NN to resolve similarly the Flicker Faces data set, but in reality each person was drastically different. I found this to be an unintended poetic and comforting discovery. When I presumed that everyone would be compressed into a conforming algorithmic data set, it appeared that uniqueness could continue. Data set 1, for example, displayed an expected appearance to training. Data set 2, partially displayed an expected appearance, but also had much less detail at some tensor levels. Data set 3 was very fussy. After several attempts it was abandoned for this project because it was not training quickly enough. Data set 4 trained the fastest and had the best Fréchet Inception Distance (FID) score of about 7. FID is a way of assessing how close a stylegan 3 generated image matches real image. Lower numbers are better. FID scores for the Flicker Face High Quality (FFHQ) is typically between 4 and 2. Data set 4 also displayed mostly hard-edged color shapes within it's NN; whereas those prior NNs would have gradient clouds.  Data set 5 was also fussy. It could generate an OK-ish image with a FID around 60, but I wanted better. A significant amount of extra training brought the FID down to 30s and improving the image quality quite a bit. Data set 5 was also the sparsest. In some tensor layers it looked like only a single color.





how a NN is both algorithm and database combined



## Effect


reinforcement learning -> really builds stereotypes

https://www.vice.com/en/article/qjk8jx/racist-algorithms-are-making-robots-racist-too

https://spectrum.ieee.org/in-2016-microsofts-racist-chatbot-revealed-the-dangers-of-online-conversation



## Conclusion

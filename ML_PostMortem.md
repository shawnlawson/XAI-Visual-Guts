# Looking Back

Going into this project I had a lot of concerns about machine learning (ML) with regards to sourcing the data used in training, curating that data for training, understanding what/how ML is doing/computing, and results were affecting people. Now at the end, those concerns are re-affirmed.

## Sourcing Data

Despite there existing a large number of open source datasets[^1], I felt it was best to capture my own data set for this project. Even if the data sets are open source, how do I know who's data it really is? Who owns it? Did they give permission? Do they even know? For example, look at some pre-built neural nets (NN)[^2]. Some of these have copyright information and some don't. Some are created from other artist's work without their knowledge. I can now create an near infinite number of paintings in the likeness of another artist's style. Is this acceptable? 

I wanted to be sure that my data was ethically sourced and collected. I wanted those people involved as data to be aware of the extent to which their image, likeness, representation could be use in machine learning. The data would go no further than me - not to be sold or available for others to use. 

Each person stood in front of a camera for a length of time between 10 and 30 min. They were asked to tell a story of 



Data Feminism

## Curating Data

Once the data was collected, as video, I had to decide how to curate the data down into chunks that could be trained. Should I only pick the most flattering? Do I worry about an even distribution of left, center, and right facing images? Should I mirror my data set to gain more information for the learning process? Do I drop the outliers - images that don't train well or disrupt the learning flow? Should I batch images in sets of 8 or 32? 

This source video was down-sampled into 1 frame per second (fps) and 10 fps sequences. The images were further resized from 16:9 aspect ratio to 1:1 (square) with a power of two width and height (1024x1024). The 1 fps sequence was used to quickly pre-train the NN and set a quick baseline. Then the 10 fps image data set was used for adding further detail into the NN. 

With that process, each person became their own unique data set rather than to sample everyone together to avoid the flicker face data set[^5] look. Moreover, rather than being overtly selective, my process was to be as inclusive as possible by not dropping any images from the image sequences. I wanted to include the outliers and difficult images, our good sides and bad, and each person's individuality/behaviors. This most likely created difficulty in training quickly or _algorithmically accurate,_ but I felt like everything was important to include.


## Understanding Data

I decided to train a separate NN per person rather than train everyone together. In this way, I would be able to more effectively make comparisons. To my surprise I found shocking differences between NNs during the training process. I had expected each data set's NN to resolve similarly the Flicker Faces data set, but in reality each person was drastically different. I found this to be an unintended poetic and comforting discovery. When I presumed that everyone would be compressed into a conforming algorithmic data set, it appeared that uniqueness could continue. Data set 1, for example, displayed an expected appearance to training. Data set 2, partially displayed an expected appearance, but also had much less detail at some tensor levels. Data set 3 was very fussy. After several attempts it was abandoned for this project because it was not training quickly enough. Data set 4 trained the fastest and had the best Fréchet Inception Distance (FID) score of about 7. FID is a way of assessing how close a stylegan 3 generated image matches real image. Lower numbers are better. FID scores for the Flicker Face High Quality (FFHQ) is typically between 4 and 2. Data set 4 also displayed mostly hard-edged color shapes within it's NN; whereas those prior NNs would have gradient clouds.  Data set 5 was also fussy. It could generate an OK-ish image with a FID around 60, but I wanted better. A significant amount of extra training brought the FID down to 30s and improving the image quality quite a bit. Data set 5 was also the sparsest. In some tensor layers it looked like only a single color.





how a NN is both algorithm and database combined



## Effect



reinforcement learning -> really builds stereotypes

https://www.vice.com/en/article/qjk8jx/racist-algorithms-are-making-robots-racist-too

https://spectrum.ieee.org/in-2016-microsofts-racist-chatbot-revealed-the-dangers-of-online-conversation



## Conclusion





[^1]: [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
[^2]: [https://github.com/justinpinkney/awesome-pretrained-stylegan2](https://github.com/justinpinkney/awesome-pretrained-stylegan2)
[^3]: By Kate Crawford and Trevor Paglen Excavating AI The Politics of Images in Machine Learning Training Sets https://excavating.ai
[^4]: joy buolamwini algorithmic justice league https://www.ajl.org/about
[^5]: [https://github.com/NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) 
[^6]: 



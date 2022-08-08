# Looking Back

Going into this project I had a lot of concerns about machine learning (ML) with regards to sourcing the data used in training, curating that data for training, understanding  how ML is computing, and what effects that computation had on people. Wrapping this all within a quantum computing wrapper that implies near unimaginable-able computation speed. Now at the end, those concerns are re-affirmed. 


## Sourcing Data

Despite there existing a large number of open source datasets[^1], I felt it was best to capture my own data set for this project. Even if the publicly data sets are open source, how do I know who's data it really is? Who owns it? Did they give permission? Do the sources of the data even know they are data? For example, looking at pre-built neural nets (NN)[^2]. Some  have copyright information and some don't. Some are created from other artist's work without their knowledge. I can now create an near infinite number of paintings in the likeness of another artist's style. Is this acceptable? Should the artist receive royalties? 

Then there is bad or incomplete data. 


Data Feminism


## Curating Data

Once the data was collected, as video, I had to decide how to curate the data down into chunks that could be trained. Should I only pick the most flattering? Do I worry about an even distribution of left, center, and right facing images? Should I mirror my data set to gain more information for the learning process? Do I drop the outliers - images that don't train well or disrupt the learning flow? Should I batch images in sets of 8 or 32? 




## Understanding Data

The trained NNs themselves describe the data set through an algorithm. 

I decided to train separate NNs for each person rather than train everyone together into a single NN. In this way, I would be able to more effectively make comparisons. To my surprise, I found shocking differences between NNs during the training process. I had expected each data set's NN to resolve similarly the Flicker Faces data set, but in reality each person was drastically different. I found this to be an unintended poetic and comforting discovery. When I presumed that everyone would be compressed into a conforming algorithmic data set, it appeared that uniqueness could continue. 

### Data set 1

Displayed an expected appearance to training - typical smooth gradients. 

### Data set 2

Partially displayed an expected appearance, but also had much less detail at some tensor levels. 

### Data set 3

Very fussy. After several attempts to train this data set, it was abandoned for this project because it was resolving quickly enough. 

### Data set 4

Trained the fastest and had the best Fréchet Inception Distance (FID) score of about 7. FID is a way of assessing how close a stylegan generated image matches real image. Lower numbers are better. FID scores for the Flicker Face High Quality (FFHQ) is typically between 4 and 2. This data also displayed mostly hard-edged color shapes within it's NN.

### Data set 5

Fussy. It could generate an OK-ish image with a FID around 60, but I wanted better. A significant amount of extra training brought the FID down to 30's and improved the image quality quite a bit. This data set 5 was also the sparsest. In some tensor layers it looked like only a single color.



## Effect



reinforcement learning -> really builds stereotypes

https://www.vice.com/en/article/qjk8jx/racist-algorithms-are-making-robots-racist-too

https://spectrum.ieee.org/in-2016-microsofts-racist-chatbot-revealed-the-dangers-of-online-conversation



## Conclusion





[^1]: [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
[^2]: [https://github.com/justinpinkney/awesome-pretrained-stylegan2](https://github.com/justinpinkney/awesome-pretrained-stylegan2)
[^3]: Kate Crawford and Trevor Paglen, "Excavating AI: The Politics of Images in Machine Learning Training Sets," https://excavating.ai
[^4]: Joy Buolamwini, et al, "Algorithmic Justice League," https://www.ajl.org/about
[^6]: 



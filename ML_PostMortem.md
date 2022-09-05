# Looking Back

Going into this project I held a lot of concerns about machine learning (ML) with regards to sourcing the data used in training, curating that data for training, understanding  how ML is computing, and what effects that computation had on people. Wrapping this all within a quantum computing wrapper that implies near unimaginable-able computation speed. Now at the end, those concerns are re-affirmed. 


## Sourcing Data

Despite there existing a large number of open source datasets[^1], I felt it was best to capture my own data set for this project. Even if the publicly data sets are open source, how do I know whose data it really is? Who owns it? Did they give permission? Do the sources of the data even know they are data? For example, looking at pre-built neural nets (NN)[^2]. Some  have copyright information and some don't. Some are created from other artist's work without their knowledge. I can now create an near infinite number of paintings in the likeness of another artist's style. Is this acceptable? Should the artist receive royalties? 

Where data comes from has an incredible impact on the final NN as discovered and critiqued by Crawford and Paglen[^3]. Moreover, insufficient data creates instances experienced racism, like that of Joy Buolamwini. This lead her to create the Algorithm Justice League[^4]. 

A method to approach many issues and questions around _data_ is through D'Ignazio and Klein's concept of Data Feminism: 

>    Throughout the book, we have described out seven principles of data feminism: examine power, challenge power, elevate emotion and embodiment, rethink binaries and hierarchies, embrace pluralism, consider context, and make labor visible. We derived these principles from the major ideas that have emerged in the past several decades of intersectional feminist activism and critical thought. At the same time, we welcome the notion that there are many other possible starting points that share the end goal of using data (or refusing data) in order to end oppression.[^5]

During the course of this project, I have tried to be aware of choices made. And, to the extent that I am able, the implications those choices have.


## Curating Data

Once the data was collected, as video, I had to decide how to curate the data down into chunks that could be trained. Should I only pick the most flattering? Do I worry about an even distribution of left, center, and right facing images? Should I mirror my data set to gain more information for the learning process? Do I drop the outliers - images that don't train well or disrupt the learning flow? Should I batch images in sets of 8 or 32? 

These were all easy questions for me to answer. Although, I became aware of how quickly they become problematic. As the curator of which images become part of the NN, consciously and unconsciously I am biased towards any selection. Even the act of choosing to not select images, I am creating some type of bias. Running through a multitude of thought experiments about how to have an unbiased data set; I settled on accepting bias will occur. Except that, when bias is noticed, I would be re-curating my data set and update my training rather than do nothing. 


## Understanding Data

I find the generative adversarial networks (GANs) computationally very interesting. First, the trained NNs are algorithms/functions; data goes in to be processed and resulting data comes out. Second, the NNs are databases (dictionaries); description (key) goes in and a discrete computed image (object) comes out. Third, the NNs are the result of compression; in my case 16GB of data was trained into a NN with a size 240MB. In summation, a NN is a compressed database and algorithm. Exact data can be recalled - like a database; but also all data in between the exact data - like a function.

I decided to train separate NNs for each person rather than train everyone together into a single NN. In this way, I would be able to more effectively make comparisons at all stages of the project. To my surprise, I found shocking differences between NNs during the training process. I had expected each data set's NN to resolve similarly to the Flicker Faces data set with nice smooth gradients and lots of detail. In reality, each person was drastically different. I found this to be an unintended, poetic, and comforting discovery. When I presumed that everyone would be compressed into a conforming algorithmic data set, it appeared that uniqueness could be preserved. 

### Data set 1

Displayed an expected appearance to training - typical smooth gradients. Although latent space navigation tended to always swirl counter-clockwise. 

### Data set 2

Partially displayed an expected appearance, but also had much less detail at some tensor levels. 

### Data set 3

Very fussy. After several attempts to train this data set, it was abandoned for this project because it was not resolving quickly enough.

### Data set 4

Trained the fastest and had the best Fréchet Inception Distance (FID) score of about 7. This data also displayed mostly hard-edged color shapes within it's NN.

### Data set 5

Fussy. It could generate an OK-ish image with a FID around 60, but I wanted better. A significant amount of extra training brought the FID down to 30's and improved the image quality quite a bit. This data set 5 was also the sparsest. In some tensor layers it looked like only a single color.


## Conclusion

The process of training a NN is that of reinforcement learning. Which makes it hard to not perceive NNs as stereotype enhancers. When I reconsider the data source and data curation process, these may be the most important stages. Similar to how advertisements work through juxtaposition of product to movie star, any intentional or unintentional juxtaposition of data in the data set will propagate and be reinforced thousands or millions of times over. A simple google search turns up quick examples: [racist chatbot](https://spectrum.ieee.org/in-2016-microsofts-racist-chatbot-revealed-the-dangers-of-online-conversation) and [racist algorithms](https://www.vice.com/en/article/qjk8jx/racist-algorithms-are-making-robots-racist-too). 

Pause. 

What happens to all of this when it intersects with the space of quantum computing? 

In the space of machine learning the easiest guess is to say that training time will become instantaneous, rather than waiting weeks and consuming incredible amounts natural resources. This is a double-sided result. Great, we can generate and test thousands of variations to achieve more accurate NNs. It's like standing on the doorstep to incredible leaps forward in technological progress and improvements to humanity. 

Not so great, we can generate and test thousands of variations to achieve more accurate NNs. Speed has always been an issue. Virilio and Lotringer have discussed the effects of speed on war[^6]. While this is not a forgone conclusion, it is critical to know how important speed is to many aspects of techno-modern human existence. As with any emergent technology, it's power to control or dominate has been a historical reality throughout human history. 

Who creates and has access to quantum computing will rapidly shift global economic and government structures. Encryption keys or passwords that would have taken millions of years to decrypt are now irrelevant. Want to spin up a DeepFake[^7] NN to propagandize, to psyops[^8] a group of people, to rewrite historical broadcasts[^9], or to deliver mass media news by anyone without consent - no problem. You can have it and be distributing what you need in a matter of seconds. Total information control. "One moves towards a centrally programmed, totalitarian society of image receivers and image administrators... ."[^10] This includes real-time broadcast and image manipulation/reconstruction. Only real life will become the verifiable ground truth. Anything else is subject to possible fabrication. 

Pause.

In this moment, I reflect on a couple of Data Feminism's principles: examine power and challenge power. Making machine learning more explainable, or being able to explain how machine learning is computing, even somewhat, returns some power back to the exploited. 

Injecting more Data Feminism Principles: embrace pluralism, consider context, and make labor visible. Making the end-to-end process of data collection, data curation, and data labeling more open and explainable allows more people to have a voice, know how data is used, and be aware of what implications varying datasets have. 

Returning to Data Feminism to consider: elevate emotion and embodiment. Too frequently data is disassociated from it's context and real world bodily impact. Everything is situational, meaning that one machine learning solution is most likely not a perfect fit for another situation. Machine learning should be used by those it directly impacts. Rather than external actors imposing their perspective without having direct, lived experience of the impact.

Finally, let's remind ourselves of the last set of Data Feminism principles: rethinking binaries and hierarchies. Quote often machine learning is used for classification tasks: male/female, white/black, and terrorist/non-terrorist. Already we see how this quickly turns into a us versus them. We also recognize that this approach completely ignores trans-gender, latino, mixed-ethnicity, and so on. Anyone not fitting a clean classification is marginalized and made invisible, thereby structurally powerless in the system.

Pause.

I think that while we brush the edge of potentially sophisticated algorithms and sit on the cusp of incredible compute speed we really should remember why we make these technological leaps and bounds; and the responsibility inherit.[^11] It should not just be for the inventors and investors; but for everyone, humanity at large. 






[^1]: [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
[^2]: [https://github.com/justinpinkney/awesome-pretrained-stylegan2](https://github.com/justinpinkney/awesome-pretrained-stylegan2)
[^3]: Kate Crawford and Trevor Paglen, "Excavating AI: The Politics of Images in Machine Learning Training Sets," https://excavating.ai
[^4]: Joy Buolamwini, et al, "Algorithmic Justice League," https://www.ajl.org/about
[^5]: Catherine D'Ignazio and Lauren F. Klein. "Data Feminism," The MIT Press, Cambridge, Massachusetts. 2020. pg 213. 
[^6]: Paul Virilio and Sylvère Lotringer, "Pure War, revised edition" Semiotext(e), 1997.
[^7]: https://en.wikipedia.org/wiki/Deepfake
[^8]: https://en.wikipedia.org/wiki/Psychological_warfare
[^9]: George Orwell, "Nineteen Eighty-Four," 1949.
[^10]: Vilém Flusser, "Into the Universe of Technical Images", University of Minnesota press, Minneapolis, 2011. pg 4.
[^11]: Historical example of incredible technological progress and responsibility, https://en.wikipedia.org/wiki/Manhattan_Project


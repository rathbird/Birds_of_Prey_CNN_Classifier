# Birds_of_Prey_CNN_Classifier
Convolutional Neural Network Classifier on two types of Birds of Prey (Eagle / Vulture)
<p>For a human, the differences between eagles and vultures are very easy to learn. Eagles have giant beaks, heavy, feathered heads, bony ridges above their eyes giving them a stern prosecutorial look, long strong wings (pound for pound, an eagle's wing is stronger than an airplane wing). If you held thirty eagle wings in your hand, they would weigh less than a penny. Then entire skeleton of a bald eagle weighs a bit more than half a pound. Vultures too have distinctive features, heavy bodies, long sinuous necks, heads with very little to no feathering, huge wings, up to 11 feet for the largest raptor of all, the Andean Condor.   </p>
<p>Their habits are also diverse. Eagles are solitary, territorial creatures. If you see two eagles together, they are a mated pair, and good luck telling the genders apart! Eagles are deadly hunters and most of the time carry their prey back to the nest to eat. Only if it's too heavy to carry will an eagle eat at the kill site. Vultures, by contrast are social birds. If you see a group of large birds hanging out together in a tree or at a carcass, you can bet it's a group of vultures. While they are feeding, they are called a wake. Vultures sometimes kill their prey, but mostly, they are scavengers, feeding on the dead carcasses left by another predator. Vultures do not have strong legs to carry meat back to their nests, so they feast at the site of the kill.</p>

## Curating Images to Train my CNN
I needed a large quantity of images for my classes, at least 1000 per class for the train set. The easiest method to gather images is to use the Flickr api. Follow these instructions if you want to build your own model:

1) Obtain a flickr api key / secret here: https://www.flickr.com/services/api/misc.api_keys.html (once downloaded, create a JSON file or copy my credentials.json file and put your values in the appropriate blanks)
2) Copy the scraper.py file in /src from https://github.com/antiboredom/flickr-scrape - I did not create a virtual environment for the scraping. using this file, you can simply run the command line from your terminal in the format: <br><i> python scraper.py --search "SEARCH TERM" --max_pages 1 </i><br>(this will give you 1000 image links)
3) This code will create an images directory if one does not exist, and then create a subdirectory based on the search term or group name used in the search, and it will place a JSON file into that directory with as many pages of images requested.
4) Open each JSON file with pandas dataframe read_json functionality. Use a list of directories and a loop of dataframes to scrape the images (see birdsofprey.ipynb in /src)
5) Curate your images! Even when using a group, which is a curated topic-centered collection, there is a lot of junk. Images fuzzy or too far away, or things that don't really match your needs, like the Philadelphia Eagles football team!

## Create an Image Pipeline
Saving memory to use in model training is critically important. I used the Tensorflow ImageDataGenerator to load images as they were being trained as opposed to loading each image into an array and storing them in memory (2000+!!!). This also provides the opportunity to expand your train set by skewing, rotating, stretching, rescaling, zooming, flipping, etc, thus expanding your trainable images. See any of the model.py files in /src for examples

<p>I really expected my transfer models trained with imagenet weigts to get me to 80-90% accuracy very quickly, but that wasn't the reality. These are the models I worked with:
  <li>Minigooglenet
    <li>InceptionV3
      <li>VGG16
        <li>Sequential
          <li>Xception
            
I ended up with most of my models overfitting, guessing eagle 100% of the time. None of my transfer models did better than random guessing. The Minigooglenet model performed the best, yet still only achieving 62% accuracy, and also prone to over-fitting. I used categorical crossentropy as my loss function and accuracy as my metric. I used softmax activation which in my final layer which is recommended for classifiers. I used various optimizers and saw negligible difference between them, SGD, RMSProp, Adam. Initially I would train 20-50 epochs with a learning rate of .0001 (default), and then do an additional training with 20 epochs with a faster learning rate, like .001 or .0005. I had middling success with these strategies.
           
I do plan to continue my quest to produce a better CNN model. The options for improving my model are 1) More images, 2) Lower learning rate, 3) Additional Layers / Additional trainable layers, and 4) Additonal dropout layers or a larger dropout rate. I found this source to be useful in getting me started with CNN classification models: https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/
            
## Building a Flask Website
All code is located in the 

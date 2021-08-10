def get_file_paths(path):
    '''
    Takes a folder path and returns a set of all file paths of .jpg in the folders
    Input: Folder path
    '''
    file_set = set()

    for direct, _, files in os.walk(path):
        for file_name in files:
            rel_dir = os.path.relpath(direct, path)
            rel_file = os.path.join(rel_dir, file_name)
            if '.jpg' not in rel_file:
                continue
            file_set.add(str(path)+rel_file)

    return file_set

def one_hotify(y, n_classes=None):
    '''Convert array of integers to one-hot format;
    The new dimension is added at the end.'''
    if n_classes is None:
        n_classes = max(y) + 1
    labels = np.arange(n_classes)
    y = y[..., None]
    return (y == labels).astype(int)

def load_images(path,size=(100,100),target='not'):
    
    file_paths = get_file_paths(path)
    
    images = []
    y = []
    for file in file_paths:
        img = keras.preprocessing.image.load_img(file, target_size=size)
        img_arr = keras.preprocessing.image.img_to_array(img)
        images.append(img_arr)
        if target in file.split('/')[-2]:
            y.append(1)
        else:
            y.append(0)
        
    return images, pd.get_dummies(y)

def change_trainable_layers(model, trainable_index):
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True
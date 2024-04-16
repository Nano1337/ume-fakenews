import os
import pandas as pd

data_path = "../data/weibo/"
corpus_dir = os.path.join(data_path, "tweets")

data_type = "train"

rumor_content = open('{}/{}_rumor.txt'.format(corpus_dir, data_type)).readlines()
nonrumor_content = open('{}/{}_nonrumor.txt'.format(corpus_dir, data_type)).readlines()

data = []  # List to store the data

nonrumor_image_path = "/home/haoli/Documents/ume-fakenews/data/weibo/nonrumor_images"
rumor_image_path = "/home/haoli/Documents/ume-fakenews/data/weibo/rumor_images"


counter = 0
# process rumor_content
n_lines = len(rumor_content)
for idx in range(2, n_lines, 3):
    one_rumor = rumor_content[idx].strip()
    if one_rumor:
        image = rumor_content[idx-1].split('|')[0]
        image = image.split('/')[-1].split('.')[0]
        # check if image exists
        if not os.path.exists(os.path.join(rumor_image_path, image+".jpg")):
            counter += 1
            # print("Image not found: ", image)
            continue
        data.append({"img_idx": image, "text": one_rumor, "label": 0})

print("Number of missing images in rumor: ", counter)
        
counter = 0

n_lines = len(nonrumor_content)
for idx in range(2, n_lines, 3):
    one_rumor = nonrumor_content[idx].strip()
    if one_rumor:
        image = nonrumor_content[idx-1].split('|')[0]
        image = image.split('/')[-1].split('.')[0]
        if not os.path.exists(os.path.join(nonrumor_image_path, image+".jpg")):
            counter += 1
            # print("Image not found: ", image)
            continue
        data.append({"img_idx": image, "text": one_rumor, "label": 1})

print("Number of missing images in nonrumor: ", counter)

# Create a DataFrame
df = pd.DataFrame(data)

# save as csv
df.to_csv('{}/{}_data.csv'.format(corpus_dir, data_type), index=False)
# print(df.head())
# print(df.shape)
# print(df.label.value_counts())
# print(df.tail())
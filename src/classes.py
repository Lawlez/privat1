import requests

url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
labels_text = requests.get(url).text.strip().splitlines()

print("Number of labels:", len(labels_text))  # should be 1001

# Let's see which line contains 'keeshond'
for i, label in enumerate(labels_text):
    # Make it lowercase for a quick substring check
    if "pizza" in label.lower():
        print(i, label)

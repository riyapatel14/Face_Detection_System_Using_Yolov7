from roboflow import Roboflow
rf = Roboflow(api_key="pushUxcO4QD23l8yStGo")
project = rf.workspace().project("face-detection-oushq")
model = project.version(1).model

# infer on a local image
print(model.predict("1.jpg", confidence=40).json())

# visualize your prediction
model.predict("1.jpg", confidence=40).save("2.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
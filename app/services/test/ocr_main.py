import easyocr

reader = easyocr.Reader(['en'])

results = reader.readtext("datasets/mediclaim/1 (1).jpg")

for bbox, text, confidence in results:
    print(text, confidence)

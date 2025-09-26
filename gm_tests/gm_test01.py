from focoos import ModelManager


im = "https://public.focoos.ai/samples/motogp.jpg"  # can be local/remote path, np.array, PIL image

model = ModelManager.get("fai-detr-l-obj365")  # any models from ModelRegistry, FocoosHub or local folder

detections = model.infer(im, annotate=True)

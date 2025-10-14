

if __name__ == "__main__":
    import numpy
    import pickle
    import zipfile


    with zipfile.ZipFile("record_0.zip", "r") as file:
        data = pickle.loads(file.read("data.pkl"))

        print("Read", len(data), "snapshotwas")
        print(data[0].image)
        print([e.__dict__ for e in data])

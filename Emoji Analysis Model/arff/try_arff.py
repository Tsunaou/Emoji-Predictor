import arff

if __name__ == '__main__':
    data = [
        [1.2, 2, 3, 1],
        [1, 2.5, 3, 2],
        [1, 2, 2.3, 3],
        [1, 2.2, 3, 1],
        [1.2, 2.2, 3, 2],
        [1, 2.8, 3, 3],
    ]

    print(arff.dumps(data))
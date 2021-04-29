import numpy as np

picture_main = [50, 23, 10, 17]

sample_image_1 = {'percents': [70, 30], 'feature': 0.6, 'distances': [[90, 10, 20, 30],
                                                                      [10, 40, 50, 40]]}

sample_image_2 = {'percents': [30, 30, 30, 10], 'feature': 1.2, 'distances': [[60, 50, 60, 60],
                                                                              [40, 60, 40, 40],
                                                                              [40, 60, 40, 40],
                                                                              [40, 60, 40, 40]]}

sample_image_3 = {'percents': [10, 50, 40], 'feature': 0.9, 'distances': [[10, 5, 90, 60],
                                                                          [40, 30, 20, 10],
                                                                          [17, 60, 10, 5]]}

sample_image_4 = {'percents': [100], 'feature': 0.8, 'distances': [[10, 20, 50, 30]]}


def calculating_distances(main_picture, sample):
    distance, percent = [0, 0]
    main_picture = np.asarray(main_picture)
    sample_feature = np.power(np.tanh(sample['feature']), 2) * 120000
    sample_percent = np.asarray(sample['percents'])
    minimum_value = np.amin(sample['distances'])
    if minimum_value > 20:
        print("Too Much Difference!\n")
    else:
        distance = np.sum(main_picture * np.power(sample['distances'], 2))
        for i in range(0, len(main_picture)):
            for j in range(0, len(sample_percent)):
                percent += (sample_percent[j] / main_picture[i]) * (sample_percent[j] - main_picture[i]) ** 2
        return distance + percent + sample_feature


first_image = calculating_distances(picture_main, sample_image_1)
second_image = calculating_distances(picture_main, sample_image_2)
third_image = calculating_distances(picture_main, sample_image_3)
fourth_image = calculating_distances(picture_main, sample_image_4)

print('First Picture:', first_image)
print('Second Picture:', second_image)
print('Third Picture:', third_image)
print('Fourth Picture:', fourth_image)

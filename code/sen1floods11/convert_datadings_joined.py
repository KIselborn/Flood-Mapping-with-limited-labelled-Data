from convert_datadings_seperately import *

def JoinedSen1Floods11Data(f_name: str, data: Union[Tuple[np.ndarray, ...]], is_weak: bool, with_dem: bool):
    key, region = key_and_region(f_name)
    s1_image = data[0]
    s2_image = data[1]
    index = 2
    if is_weak:
        label = None
        s1_weak_label_image = data[index]
        s2_weak_label_image = data[index+1]
        index+=2
    else:
        label = data[index]
        s1_weak_label_image = None
        s2_weak_label_image = None
        index+=1
    dem = data[index] if with_dem else None
    res = {
        'key': key,
        'region': region,
        's1_image': s1_image,
        's2_image': s2_image
    }
    if label is not None:
        res['label_image'] = label
    if s1_weak_label_image is not None and s2_weak_label_image is not None:
        res['s1_weak_label_image'] = s1_weak_label_image
        res['s2_weak_label_image'] = s2_weak_label_image
    if dem is not None:
        res['dem_image'] = dem
    return res


def convert_joined_to_datadings(base_folder: str, output_folder: str):
    pass
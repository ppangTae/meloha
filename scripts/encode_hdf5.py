import os
from meloha.utils import compress_hdf5  # 실제로 compress_hdf5가 정의된 파일명에 맞게 바꿔야 함

def main():
    dataset_dir = "/home/graduation_ann/meloha_data/meloha_box_picking"
    filenames = sorted(os.listdir(dataset_dir))

    for name in filenames:
        if not name.endswith(".hdf5"):
            continue
        if "_compressed" in name:
            continue  # 이미 압축된 파일은 건너뜀

        dataset_name = name.replace(".hdf5", "")
        print(f"[*] Compressing: {dataset_name}")
        try:
            compress_hdf5(dataset_dir, dataset_name)
        except ValueError as e:
            print(e)  # 이미 압축된 경우 출력
        except Exception as e:
            print(f"[!] Failed to compress {dataset_name}: {e}")

if __name__ == '__main__':
    main()

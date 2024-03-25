from dataset.convert_datasets_to_hf import get_dataset

def main():
    
    #重新处理数据并获得hf格式的数据集
    raw_dataset = get_dataset()
    
    #测试用
    print(raw_dataset[666]['content'])




if __name__ == "__main__":
    main()
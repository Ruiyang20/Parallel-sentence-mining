import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def args_parser():
    parser = argparse.ArgumentParser(description='Visualize CBIE-enhanced embeddings')

    parser.add_argument('-m', '--model_name', type=str, required=True, help='Embedding model name (e.g., glot500, xlmr)')
    
    parser.add_argument('--cbie', action='store_true', help='Enable CBIE-enhanced embeddings')

    parser.add_argument('--ori', action='store_true', help='Original data without transformation')


    return parser.parse_args()

def load_vec_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    ids = []
    vecs = []

    for line in lines[1:]:  # 跳过第一行 header
        parts = line.strip().split()
        ids.append(parts[0])
        vecs.append([float(x) for x in parts[1:]])

    return ids, np.array(vecs)

def plot_embeddings(hsb_vecs, de_vecs, title="Language Pair", save_path=None):
    # 合并向量和标签
    X = np.vstack([hsb_vecs, de_vecs])
    labels = ['hsb'] * len(hsb_vecs) + ['de'] * len(de_vecs)
    
    # PCA降维
    pca = PCA(n_components=2).fit(X)
    hsb_pca = pca.transform(hsb_vecs)
    de_pca = pca.transform(de_vecs)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(hsb_pca[:, 0], hsb_pca[:, 1], 
                s=5, c='green', alpha=0.5, label='HSB')
    plt.scatter(de_pca[:, 0], de_pca[:, 1], 
                s=5, c='skyblue', alpha=0.5, label='DE')

    plt.title(title, fontsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ 图像已保存：{save_path}")
    else:
        plt.show()

def main():
    # 替换为你的实际路径
    args = args_parser()
    model= args.model_name
    cbie = args.cbie
    ori = args.ori
    path ='./CBIE/cs-de'
    if cbie:
        hsb_path = f'{path}/CBIE2_{model}.cs-de.train.cs.vec'
        de_path = f'{path}/CBIE2_{model}.cs-de.train.de.vec'
        save_path = f'{path}/cbie_plot_{model}.png'
    else:
        # 如果没有CBIE转换，使用原始路径
        hsb_path = f'{path}/{model}.cs-de.train.cs.vec'
        de_path = f'{path}/{model}.cs-de.train.de.vec'
        save_path = f'{path}/original_plot_{model}.png'


    hsb_ids, hsb_vecs = load_vec_file(hsb_path)
    de_ids,  de_vecs  = load_vec_file(de_path)
    plot_embeddings(hsb_vecs, de_vecs, title="HSB vs DE", save_path=save_path)


if __name__ == '__main__':

    main()
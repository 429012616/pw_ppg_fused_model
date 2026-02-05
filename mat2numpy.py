import numpy as np
from scipy.io import loadmat

if __name__ == '__main__':
	mat_data = loadmat("./db_data/db_data.mat")

	print(mat_data.keys())

	db_data_non_overlap = mat_data['db_data']
	db_data_overlap = mat_data['db_data_overlap']

	db_label_non_overlap = mat_data['db_label']
	db_label_overlap = mat_data['db_label_overlap']

	db_S =  mat_data['db_S']
	db_S_n = mat_data['db_S_n']

	np.savez("./db_data/db_data.npz", db_data_non_overlap=db_data_non_overlap, db_label_non_overlap=db_label_non_overlap,db_data_overlap =db_data_overlap,db_label_overlap = db_label_overlap,db_S = db_S,db_S_n = db_S_n )
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


whisky = pd.read_csv("whiskies.txt")
whisky["Region"] = pd.read_csv("regions.txt")
# whisky.head()
# whisky.tail()


# index DataFrame by location
# print(whisky.iloc[0:10])

whisky.iloc[5:10,0:5]  #.iloc[rows, columns]
whisky.columns


#extract flavors from the table
flavors = whisky.iloc[:, 2:14]
# print(flavors)


corr_flavors = pd.DataFrame.corr(flavors)	#correlation   
# print(corr_flavors)

plt.figure(figsize=(10,10))					#plt 
plt.pcolor(corr_flavors)
plt.colorbar()
plt.savefig("corr_flavors.pdf")


corr_whisky = pd.DataFrame.corr(flavors.transpose())
plt.figure(figsize=(10,10))					
plt.pcolor(corr_whisky)
plt.colorbar()
plt.savefig("corr_whisky.pdf")



from  sklearn.cluster.bicluster import SpectralCoclustering


model = SpectralCoclustering(n_clusters=6, random_state=0) # step1
model.fit(corr_whisky)    								   # step2
model.rows_								#the number of row clusters x number of rows



"""
4. Comparing Correlation Matrices
- og: corr_whisky
- new: correlations
"""
whisky['Group'] = pd.Series(model.row_labels_, index=whisky.index)
# reorder the rows in increasing order by group labels
whisky = whisky.ix[np.argsort(model.row_labels_)]
# reset the index of DF
whisky = whisky.reset_index(drop=True)


correlations = pd.DataFrame.corr(whisky.iloc[:,2:14].transpose())
correlations = np.array(correlations)


plt.figure(figsize=(14,7))					
plt.subplot(121)
plt.pcolor(corr_whisky)
plt.title("Original")
plt.axis("tight")
plt.subplot(122)
plt.pcolor(correlations)
plt.title("Rearranged")
plt.axis("tight")
plt.savefig("correlations.pdf")





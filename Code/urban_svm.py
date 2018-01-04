
import numpy as np 
from sklearn.svm import SVC,LinearSVC
from sklearn.utils import shuffle

y = np.load('sounds_id.npy')
X = np.load('mfccs.npy')
test = np.load('mfcc_mix.npy')
tst_label = np.load("labelsmix.npy")
norm = np.amax(np.absolute(X))
tnorm = np.amax(np.absolute(test))
X = X/norm
T = test/tnorm


#l1,l2 = shuffle(X,y)
    
c = 50000

classes = ["air_conditioner","car_horn","children playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music"]    
clf = SVC(C=c,class_weight = "balanced",decision_function_shape = "ovr",gamma = .7)
    #clf1 =SVC(C=c,class_weight = "balanced",decision_function_shape = "ovr",kernel ="linear")
#clf =SVC(C=c,class_weight = "balanced",decision_function_shape = "ovr",kernel ="poly",degree= 10,coef0 = 1.205)
#clf3 =SVC(C=c,class_weight = "balanced",decision_function_shape = "ovr",kernel ="sigmoid",gamma = 0.005,coef0 = .1)

    #clf.fit(l1[:8000],l2[:8000])
    #clf1.fit(l1[:8000],l2[:8000])
    #clf2.fit(l1[:8000],l2[:8000])
clf.fit(X,y)

    #pred = clf.predict(l1[8000:])
    #pred1 = clf1.predict(l1[8000:])
    #pred2 = clf2.predict(l1[8000:])
pred3 = clf.predict(T)

    #print("Accuracy for SVM with decision function ovr and kernel rbf and c = %d: "%c,np.mean(pred == l2[8000:]))
    #print("Accuracy for SVM with decision function ovr and kernel linear: ",np.mean(pred1 == l2[8000:]))
    #print("Accuracy for SVM with decision function ovr and kernel poly: ",np.mean(pred2 == l2[8000:]))
#print("Accuracy for SVM with decision function ovr and kernel sigmoid and c = %d: "%c,np.mean(pred3 == l2[8000:]))
'''
ac = 0
for i,j in zip(pred3,tst_label):
    if(i == (int)(j) ):
        ac +=1
print(ac/200)
'''
for i,j in zip(pred3,tst_label):
    print(classes[i],j)
import numpy as np
import matplotlib.pyplot as plt

Cvalues = np.array([1e-05,1e-03,1,5,10])
kfold = np.array([9.293748240349705, 9.293748240349705, 87.88835049764145, 88.25726210762885, 88.23947346323875])
test = np.array([53.39067813562713, 53.39067813562713, 88.07761552310463, 88.27765553110622, 88.2376475295059])

plt.plot(np.log(Cvalues),kfold , label='kfold')
plt.plot(np.log(Cvalues),test , label='test')
plt.xlabel('log values of C')
plt.ylabel('accuracy')
plt.legend()
plt.show()
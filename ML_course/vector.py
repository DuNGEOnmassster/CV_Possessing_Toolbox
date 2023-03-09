import matplotlib.pyplot as plt
import numpy as np
 
x = np.linspace(0, 2*np.pi, 100)
 
y = np.sin(x**2)
 
plt.title("Sin Function")
plt.plot(x, y)
plt.show()
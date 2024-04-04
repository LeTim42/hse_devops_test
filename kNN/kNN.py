import numpy as np
from collections import Counter

class KNN:
    elems = []
    def __init__(self, k:int):
        self.k = k

    def fit(self, X, y):  #Обучение заключается в запоминании обучающей выборки(а именно признаков объекта и класса, к которому он принадлежит)
      for x_row,y_row in zip(X,y):
        self.elems.append(Counter(x=x_row[0],y=x_row[1],cat = y_row)) #В х и у сохраняем координаты нашей точки, а в cat её категорию

    def predict(self, X):
      ans = [] #Этот список будет нашим выходным списком target размера X
      for coord in X: #Для каждой точки из данной выборки сортируем список наших точек обучающей выборки по расстоянию до данной
        self.elems.sort(key=lambda x: self.count_distance(x,coord))
        cnt = Counter()
        for i in range (self.k): #Идём по первым k элементам нашего отсортированного списка(к - число соседей)
          cnt[self.elems[i]["cat"]]+=1  #Смотрим категорию i-ого элемента и по этой категории делаем +1 в нашем счётчике(словаре, по сути)
        ans.append(cnt.most_common(1)[0][0]) #most_common выводит самое популярное значение в словаре(но оно выводится в качестве массива в кортеже, поэтому надо 2 раза обратиться к нулевому элементу)
      return np.array(ans)

    def count_distance(self, x, y): #Расстояние по Евклидову расстоянию
      return ((x["x"]-y[0])**2 + (x["y"]-y[1])**2)

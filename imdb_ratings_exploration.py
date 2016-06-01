import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/imdb_1000.csv')
cols = data.columns
desc = data.describe()
# print(cols)
print(desc)

genre_pivot = data.pivot_table(index='genre',values=['star_rating','duration'], aggfunc=np.mean)
print(genre_pivot)

width = 0.35
indx = np.arange(len(genre_pivot.index.tolist()))
fig_gen = plt.figure(figsize=(16,8))
ax_star_gen = fig_gen.add_subplot(1,2,1)
ax_duration_gen = fig_gen.add_subplot(1,2,2)

ax_star_gen.set(xticks=(indx+(width/2)),xticklabels=genre_pivot.index,ylim=(min(genre_pivot['star_rating'])-0.5,max(genre_pivot['star_rating'])+0.5),title='Star Rating', xlabel='Genre', ylabel='Rating')
ax_star_gen.bar(indx,genre_pivot['star_rating'],width=0.35, color='blue')
ax_star_gen.plot(indx,[np.mean(genre_pivot['star_rating']) for i in indx],color = 'red')

xlabels = ax_star_gen.get_xticklabels() 
for label in xlabels: 
    label.set_rotation(90)

ax_duration_gen.set(xticks=(indx+(width/2)),xticklabels=genre_pivot.index,ylim=(min(genre_pivot['duration'])-10,max(genre_pivot['duration'])+10),title='Duration', xlabel='Genre', ylabel='Duration')
ax_duration_gen.bar(indx,genre_pivot['duration'],width=0.35, color='green')
ax_duration_gen.plot(indx,[np.mean(genre_pivot['duration']) for i in indx],color = 'red')

xlabels = ax_duration_gen.get_xticklabels() 
for label in xlabels:
    label.set_rotation(90)

plt.show()
____________________________________________________________________

fig_cont = plt.figure(figsize=(8,8))
ax_cont = fig_cont.add_subplot(1,1,1)

content_data = data[['content_rating','star_rating']]

ax_cont.boxplot(content_data, by='content_rating')

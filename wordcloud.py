#wordlcouds
from wordcloud import WordCloud
names = kiva_loans_data["country"][~pd.isnull(kiva_loans_data["country"])]
#print(names)
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for country Names", fontsize=35)
plt.axis("off")
plt.show() 
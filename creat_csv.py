import pandas as pd

col = ['img', 'healthy', 'leaf_rust', 'powdery_mildew', 'seedlings', 'septoria', 'stem_rust', 'yellow_rust']

df = pd.DataFrame(columns=col)

df.to_csv('./img_40/test_40.csv', index=False)
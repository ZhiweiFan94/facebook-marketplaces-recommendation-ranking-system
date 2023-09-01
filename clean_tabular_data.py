##########connect to EC2 and download files#############
#%%load files from EC2
import paramiko

# Configuration
key_file_path = "c647f700-e161-47fa-8b77-98252d959902_9f1749cc-4ba8-4037-ade0-414fcadf848b.pem"
ec2_ip_address = "ec2-34-244-8-215.eu-west-1.compute.amazonaws.com"
ec2_username = "ec2-user"
remote_file_path = "images_fb.zip"
local_file_path = "/Users/fanzhiwei/Desktop/Aicore-test/facebook-marketplaces-recommendation-ranking-system/images_fb.zip"  
# Establish SSH connection
key = paramiko.RSAKey(filename=key_file_path)
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(ec2_ip_address, username=ec2_username, pkey=key)

# Download file from EC2 to local
sftp_client = client.open_sftp()
sftp_client.get(remote_file_path, local_file_path)

# Cleanup
sftp_client.close()
client.close()


#######Clean tabular dataset and extract labels for classification#########
# %% clean products.csv 
import pandas as pd
# read products dataset
df_pdt = pd.read_csv('Products.csv',lineterminator='\n')
# check null values
print(df_pdt.isnull().sum())
# strip currency sign from the price
df_pdt['price'] = df_pdt['price'].str.strip('£')
df_pdt['price'] = df_pdt['price'].str.replace(',','').astype(float)
df_pdt.head()

# %% clean images.csv
df_img = pd.read_csv('Images.csv')
df_img.isnull().sum()

# %% extract labels of category from product dataset
# shorten the root category by first content before slash
df_pdt['ext_category'] = df_pdt['category'].str.split('/').str[0].str.strip()
category = df_pdt['ext_category'].unique()
# custom labels for each category
custom_encoding = {
    'Home & Garden': 0,
    'Baby & Kids Stuff': 1,
    'DIY Tools & Materials': 2,
    'Music, Films, Books & Games': 3,
    'Phones, Mobile Phones & Telecoms': 4,
    'Clothes, Footwear & Accessories': 5,
    'Other Goods': 6,
    'Health & Beauty': 7,
    'Sports, Leisure & Travel': 8,
    'Appliances': 9,
    'Computers & Software': 10,
    'Office Furniture & Equipment': 11,
    'Video Games & Consoles': 12
}
# create new column with encoded labels 
df_pdt['encoded_category'] = df_pdt['ext_category'].map(custom_encoding)

# %%test the decoder if category numbers are given
coder_list = [0,2,4,6,9,12]
reversed_pair = {v: k for k, v in custom_encoding.items()}
decoded_category = list(map(reversed_pair.get, coder_list))

# %% merge two dataframes with matched ids
merged_df = df_img.merge(df_pdt, left_on='product_id', right_on='id', how='inner')
# save the merged dataset 
merged_df.to_csv('training_data.csv')


import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from tqdm import tqdm

# تحميل البيانات
df = pd.read_csv("sampled_dataset.csv")

# ماب 8-Classes
dict_8classes = {
    'DDoS-RSTFINFlood': 'DDoS', 'DDoS-PSHACK_Flood': 'DDoS', 'DDoS-SYN_Flood': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS', 'DDoS-TCP_Flood': 'DDoS', 'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS', 'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS', 'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-SlowLoris': 'DDoS', 'DDoS-HTTP_Flood': 'DDoS',
    'DoS-UDP_Flood': 'DoS', 'DoS-SYN_Flood': 'DoS', 'DoS-TCP_Flood': 'DoS', 'DoS-HTTP_Flood': 'DoS',
    'Mirai-greeth_flood': 'Mirai', 'Mirai-greip_flood': 'Mirai', 'Mirai-udpplain': 'Mirai',
    'Recon-PingSweep': 'Recon', 'Recon-OSScan': 'Recon', 'Recon-PortScan': 'Recon',
    'VulnerabilityScan': 'Recon', 'Recon-HostDiscovery': 'Recon',
    'DNS_Spoofing': 'Spoofing', 'MITM-ArpSpoofing': 'Spoofing',
    'BrowserHijacking': 'Web', 'Backdoor_Malware': 'Web', 'XSS': 'Web',
    'Uploading_Attack': 'Web', 'SqlInjection': 'Web', 'CommandInjection': 'Web',
    'DictionaryBruteForce': 'BruteForce',
    'BenignTraffic': 'Benign',
}

#  
df['label'] = df['label'].map(dict_8classes)
df = df.dropna(subset=['label'])

#
print("📊 توزيع الكلاسات الأصلي:")
class_counts = Counter(df['label'])
print(class_counts)

# UnderSampling للكلاسات الكبيرة
frames = []
for cls in tqdm(class_counts, desc="UnderSampling"):
    cls_df = df[df['label'] == cls]
    if class_counts[cls] > 200_000:
        cls_df = cls_df.sample(n=200_000, random_state=42)
    frames.append(cls_df)

df_balanced = pd.concat(frames)
print("\n📉 بعد UnderSampling:")
print(Counter(df_balanced['label']))

# فصل X و y
X = df_balanced.drop(columns=['label'])
y = df_balanced['label']

# 👇 
smote_target = {
    'DoS': 100_000,
    'DDoS': 100_000,
    'Mirai': 100_000,
    'Recon': 80_000,
    'Spoofing': 60_000,
    'Web': 30_000,
    'BruteForce': 20_000,
    'Benign': 100_000
}

# تطبيق SMOTE
print("\n🔁 جاري تطبيق SMOTE...")
smote = SMOTE(sampling_strategy=smote_target, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# توزيع نهائي
print("\n✅ توزيع بعد SMOTE:")
print(Counter(y_resampled))

# حفظ الناتج في ملف
df_final = pd.DataFrame(X_resampled)
df_final['label'] = y_resampled
df_final.to_csv("balanced_dataset.csv", index=False)
print("\n📁 تم حفظ الملف باسم 'balanced_dataset.csv'")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib


model = joblib.load(r"D:\data grad\wataiData\csv\CICIoT2023\output\rf final high recall\model cross validt\rf_model_final2.pkl")
# غيّر الاسم حسب ملفك

X_columns = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
    'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count',
    'syn_count', 'fin_count', 'urg_count', 'rst_count', 
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
    'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
    'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
    'Radius', 'Covariance', 'Variance', 'Weight', 
]
y_column = 'label'

X_test = df[X_columns].values
y_test_raw = df[y_column].values

# === 2. تحويل التصنيفات التفصيلية إلى 8 تصنيفات رئيسية ===
dict_7classes = {
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
    'BenignTraffic': 'Benign',
    'BrowserHijacking': 'Web', 'Backdoor_Malware': 'Web', 'XSS': 'Web',
    'Uploading_Attack': 'Web', 'SqlInjection': 'Web', 'CommandInjection': 'Web',
    'DictionaryBruteForce': 'BruteForce'
}

y_test = [dict_7classes.get(label, 'Unknown') for label in y_test_raw]
feature_names = X_columns
class_names = ['Benign', 'BruteForce', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web']

# === 3. تحليل قرار الموديل لكل sample ===
sample_index = 0  # غيّره لو عايز تختار sample تانية
sample = X_test[sample_index].reshape(1, -1)
print(f"\n🔍 Sample {sample_index} - True Label: {y_test[sample_index]}")

for i, tree in enumerate(model.estimators_):
    leaf_id = tree.apply(sample)
    node_indicator = tree.decision_path(sample)
    print(f"\n🌲 Tree {i}:")
    print(f"  Leaf node: {leaf_id[0]}")
    print(f"  Decision path node indices: {node_indicator.indices}")

# === 4. Visualization لشجرة معينة ===
tree_index = 0  # غيّره لو عايز ترسم شجرة تانية
plt.figure(figsize=(20, 10))
plot_tree(model.estimators_[tree_index],
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          rounded=True,
          impurity=False)
plt.title(f"Decision Tree {tree_index}")
plt.show()

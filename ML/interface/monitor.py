from flask import Flask, render_template, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import subprocess
import os
from datetime import datetime
import ipaddress

app = Flask(__name__)
CORS(app)

# Load model
print("Loading model...")
model = joblib.load("rf_model_final2.pkl")
scaler = joblib.load("scaler3.pkl")
label_encoder = joblib.load("label_encoder3.pkl")
print("✅ Model loaded!")

# Feature names (48 features)
FEATURES = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
    'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count',
    'syn_count', 'fin_count', 'urg_count', 'rst_count', 
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
    'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
    'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
    'Radius', 'Covariance', 'Variance', 'Weight'
]

CLASSES = ['Benign', 'BruteForce', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web']

def get_wifi_interface():
    """Get WiFi interface name"""
    tshark = r"C:\Program Files\Wireshark\tshark.exe"
    
    try:
        # List all interfaces
        print("\n🔍 Looking for WiFi interfaces...")
        result = subprocess.run([tshark, "-D"], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("\n📡 Available network interfaces:")
            print("-" * 50)
            print(result.stdout)
            print("-" * 50)
            
            lines = result.stdout.strip().split('\n')
            wifi_interfaces = []
            
            for line in lines:
                # Look for WiFi/Wireless interface (case insensitive)
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in ['wi-fi', 'wifi', 'wireless', '802.11', 'wlan']):
                    # Extract interface number (e.g., "1. \Device\...")
                    interface_num = line.split('.')[0].strip()
                    print(f"✅ Found WiFi interface: {line}")
                    wifi_interfaces.append((interface_num, line))
            
            if not wifi_interfaces:
                print("❌ No WiFi interfaces found! Please check your WiFi connection.")
                print("Trying to find any available interface...")
                # If no WiFi interface found, try to use the first available one
                if lines:
                    interface_num = lines[0].split('.')[0].strip()
                    print(f"⚠️  No WiFi interface found. Using the first available interface: {lines[0]}")
                    return interface_num
                return None
                
            # Use the first WiFi interface found
            interface_num, interface_name = wifi_interfaces[0]
            print(f"\n🔌 Selected WiFi interface: {interface_name}")
            return interface_num
            
            # If no WiFi found, return first interface
            if lines:
                interface_num = lines[0].split('.')[0].strip()
                print(f"Using first interface: {lines[0]}")
                return interface_num
                
    except Exception as e:
        print(f"Error getting interfaces: {e}")
    
    return None

def capture_packets():
    """Capture packets using tshark from WiFi and return raw lines"""
    tshark = r"C:\Program Files\Wireshark\tshark.exe"
    
    if not os.path.exists(tshark):
        return None, "Wireshark not found"
    
    try:
        # Get WiFi interface
        interface = get_wifi_interface()
        
        if not interface:
            return None, "No WiFi interface found"
        
        cmd = [
            tshark,
            "-i",
            interface,
            "-a",
            "duration:10",
            "-T",
            "fields",
            "-e",
            "frame.len",
            "-e",
            "ip.proto",
            "-e",
            "tcp.flags.syn",
            "-e",
            "tcp.flags.ack",
            "-e",
            "tcp.flags.fin",
            "-e",
            "tcp.flags.reset",
            "-e",
            "tcp.flags.push",
            "-e",
            "tcp.dstport",
            "-e",
            "tcp.srcport",
            "-e",
            "udp.dstport",
            "-e",
            "ip.src",
            "-e",
            "ip.dst",
            "-E",
            "separator=,",
        ]
        
        print(f"📡 Capturing from WiFi interface #{interface}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
        
        if result.returncode == 0 and result.stdout:
            packets = result.stdout.strip().split("\n")
            packets = [p for p in packets if p.strip()]  # Remove empty lines
            print(f"✅ Captured {len(packets)} packets")
            # Debug: show first few lines to verify fields (including IPs)
            for idx, line in enumerate(packets[:5]):
                print(f"   sample[{idx}]: {line}")
            return packets, None
        
        # Check for errors
        if result.stderr:
            print(f"❌ tshark error: {result.stderr}")
            return None, f"Capture error: {result.stderr[:100]}"
        
        return None, "No packets captured"
        
    except subprocess.TimeoutExpired:
        return None, "Capture timeout"
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None, str(e)

def _is_public_ipv4(ip_str: str) -> bool:
    """Return True if ip_str is a valid *public* IPv4 address."""
    if not ip_str:
        return False
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        # Only IPv4 and not private / loopback / link-local / multicast / reserved
        return ip_obj.version == 4 and not (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
        )
    except ValueError:
        return False


def extract_features(packets):
    """Extract ML features and collect unique public IPv4s (src/dst) from raw tshark lines."""
    tcp = udp = syn = ack = fin = rst = psh = 0
    http = https = dns = ssh = 0
    sizes = []
    public_ips = []
    seen_ips = set()
    
    for line in packets:
        if not line.strip():
            continue
        
        fields = line.split(",")
        
        # Size
        try:
            sizes.append(int(fields[0]) if fields[0] else 0)
        except:
            sizes.append(0)
        
        # Protocol
        proto = fields[1] if len(fields) > 1 else ""
        
        if proto == "6":  # TCP
            tcp += 1
            if len(fields) > 2 and fields[2] == "1": syn += 1
            if len(fields) > 3 and fields[3] == "1": ack += 1
            if len(fields) > 4 and fields[4] == "1": fin += 1
            if len(fields) > 5 and fields[5] == "1": rst += 1
            if len(fields) > 6 and fields[6] == "1": psh += 1
            
            try:
                port = int(fields[7]) if len(fields) > 7 and fields[7] else 0
                if port == 80: http += 1
                elif port == 443: https += 1
                elif port == 22: ssh += 1
            except:
                pass
                
        elif proto == "17":  # UDP
            udp += 1
            try:
                if len(fields) > 9 and fields[9] == "53":
                    dns += 1
            except:
                pass

        # IP addresses (indices 10 and 11 according to tshark fields above)
        try:
            src_ip = fields[10] if len(fields) > 10 else ""
            dst_ip = fields[11] if len(fields) > 11 else ""
        except IndexError:
            src_ip = ""
            dst_ip = ""

        for ip_str in (src_ip, dst_ip):
            if _is_public_ipv4(ip_str) and ip_str not in seen_ips:
                seen_ips.add(ip_str)
                public_ips.append(ip_str)
    
    # Build features
    count = len(packets)
    avg = np.mean(sizes) if sizes else 0
    
    feat = {
        'Tot size': sum(sizes), 'Min': min(sizes) if sizes else 0,
        'Max': max(sizes) if sizes else 0, 'AVG': avg,
        'Std': np.std(sizes) if sizes else 0,
        'TCP': 1 if tcp > 0 else 0, 'UDP': 1 if udp > 0 else 0,
        'HTTP': 1 if http > 0 else 0, 'HTTPS': 1 if https > 0 else 0,
        'DNS': 1 if dns > 0 else 0, 'SSH': 1 if ssh > 0 else 0,
        'syn_count': syn, 'ack_count': ack, 'fin_count': fin,
        'rst_count': rst, 'urg_count': 0,
        'syn_flag_number': syn, 'ack_flag_number': ack,
        'fin_flag_number': fin, 'rst_flag_number': rst,
        'psh_flag_number': psh, 'ece_flag_number': 0, 'cwr_flag_number': 0,
        'flow_duration': count * 10, 'Header_Length': avg * 0.1,
        'Protocol Type': 6 if tcp > udp else 17,
        'Duration': count * 10, 'Rate': count / 2.0,
        'Srate': count / 4.0, 'Drate': count / 4.0,
        'Number': count, 'IAT': 100.0, 'Magnitue': 50.0,
        'Radius': 25.0, 'Covariance': 10.0, 'Variance': 15.0,
        'Weight': 1.0, 'Tot sum': sum(sizes)
    }
    
    # Fill missing
    for f in FEATURES:
        if f not in feat:
            feat[f] = 0 if f in ['Telnet', 'SMTP', 'IRC', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC'] else 0.0
    
    return feat, count, public_ips

@app.route('/')
def index():
    return render_template('monitor.html')

@app.route('/interfaces')
def list_interfaces():
    """List available network interfaces"""
    tshark = r"C:\Program Files\Wireshark\tshark.exe"
    
    try:
        result = subprocess.run([tshark, "-D"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            interfaces = result.stdout.strip().split('\n')
            return jsonify({'success': True, 'interfaces': interfaces})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': 'Could not list interfaces'})

@app.route('/capture', methods=['POST'])
def capture():
    """Capture and predict"""
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Capturing...")
        
        packets, error = capture_packets()
        if error:
            return jsonify({'success': False, 'error': error})
        
        features, count, public_ips = extract_features(packets)
        
        # Predict
        X = np.array([features[f] for f in FEATURES]).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        
        result = label_encoder.inverse_transform([pred])[0]
        probs = {CLASSES[i]: float(proba[i] * 100) for i in range(len(CLASSES))}
        
        # Reduce Recon false positives
        if result == 'Recon' and probs[result] < 65 and probs['Benign'] > 20:
            result = 'Benign'
        
        print(f"✅ {result} ({probs[result]:.1f}%) - {count} packets")
        
        return jsonify({
            'success': True,
            'prediction': result,
            'probabilities': probs,
            'packets': count,
            'time': datetime.now().strftime('%H:%M:%S'),
            'public_ips': public_ips,
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("\n🚀 Starting Live Monitor on http://192.168.1.9:5001\n")
    app.run(debug=True, host='192.168.1.9', port=5001, use_reloader=False)

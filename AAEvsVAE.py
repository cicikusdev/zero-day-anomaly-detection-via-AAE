!pip install lime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt

# ==========================================
# 1. AKILLI VE DİNAMİK DATASET FACTORY (Hata Korumalı)
# ==========================================
class ModernNetworkDataset(Dataset):
    def __init__(self, csv_path, probable_label_names, normal_label_str, is_train=True):
        print(f"-> Dosya okunuyor: {csv_path} (Bu işlem dosya boyutuna göre zaman alabilir...)")
        preview = pd.read_csv(csv_path, nrows=5)
        preview.columns = preview.columns.str.strip() 
        
        actual_label_col = None
        for name in probable_label_names:
            match = [c for c in preview.columns if c.lower() == name.lower()]
            if match:
                actual_label_col = match[0]
                break
                
        if actual_label_col is None:
            raise KeyError(f"HATA: Veri setinde etiket (Label) sütunu bulunamadı! Mevcut sütunlar: {list(preview.columns[:10])}...")
            
        print(f"-> Başarılı: Etiket sütunu olarak '{actual_label_col}' tespit edildi.")
        
        self.data = pd.read_csv(csv_path)
        self.data.columns = self.data.columns.str.strip()
        
        unique_labels = self.data[actual_label_col].unique()
        print(f"-> Veri setindeki benzersiz etiketler: {unique_labels}")
        
        matched_normal_label = None
        for lbl in unique_labels:
            if str(lbl).strip().lower() == str(normal_label_str).strip().lower():
                matched_normal_label = lbl
                break
                
        if matched_normal_label is None:
            raise ValueError(f"HATA: Normal trafik için aranan '{normal_label_str}' değeri bulunamadı. Mevcut değerler: {unique_labels}")
            
        if is_train:
            self.data = self.data[self.data[actual_label_col] == matched_normal_label]
            print(f"-> Sadece normal trafik filtrelendi. Eğitim satır sayısı: {len(self.data)}")
        else:
            print(f"-> Test aşaması: Tüm trafik yüklendi. Toplam satır sayısı: {len(self.data)}")
        
        self.labels = self.data[actual_label_col].values
        features_df = self.data.drop(columns=[actual_label_col])
        
        # --- GELİŞMİŞ GEREKSİZ/ZARARLI SÜTUN TEMİZLİĞİ ---
        drop_cols = [
            'Timestamp', 'timestamp', 'date', 'time',
            'Source IP', 'Destination IP', 'src_ip', 'dst_ip',
            'Source Port', 'Destination Port', 'src_port', 'dst_port',
            'Unnamed: 0', 'id', 'Flow ID'
        ]
        for col in drop_cols:
            matches = [c for c in features_df.columns if c.lower() == col.lower()]
            if matches:
                features_df.drop(columns=matches, inplace=True)
                
        for col in features_df.select_dtypes(include=['object']).columns:
            features_df[col] = features_df[col].astype('category').cat.codes
            
        self.feature_names = features_df.columns.tolist()
        
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        self.features = self.imputer.fit_transform(features_df)
        
        self.scaler = MinMaxScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32)

class CICEncryptedTraffic2026Dataset(ModernNetworkDataset):
    def __init__(self, csv_path, is_train=True):
        # 2026 seti için normal_label_str 'HTTPS' olarak düzeltildi
        super().__init__(csv_path, probable_label_names=['label', 'Label', 'class', 'Class'], normal_label_str='HTTPS', is_train=is_train)

class CICIDS2018Dataset(ModernNetworkDataset):
    def __init__(self, csv_path, is_train=True):
        super().__init__(csv_path, probable_label_names=['Label', 'label', 'Class', 'class'], normal_label_str='Benign', is_train=is_train)

def get_dataloader(dataset_name, csv_path, batch_size, is_train=True):
    if dataset_name == 'CIC2018':
        dataset = CICIDS2018Dataset(csv_path, is_train=is_train)
    else:
        dataset = CICEncryptedTraffic2026Dataset(csv_path, is_train=is_train)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, drop_last=is_train)
    return loader, dataset


# ==========================================
# 2. MODELLER (AAE ve VAE Mimarileri)
# ==========================================
class BaseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.net(x)

class BaseDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim),
            nn.Sigmoid() 
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.net(z)

class AAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            BaseEncoder(input_dim, hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = BaseDecoder(latent_dim, hidden_dim, input_dim)
        self.discriminator = Discriminator(latent_dim)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=8):
        super().__init__()
        self.shared_encoder = BaseEncoder(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = BaseDecoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.shared_encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# ==========================================
# 3. EĞİTİM ADIMLARI
# ==========================================
def train_aae_step(model, opt_G, opt_D, batch_data):
    model.train()
    batch_size = batch_data.size(0)
    
    # 1. Reconstruction Aşaması
    opt_G.zero_grad()
    latent_z = model.encoder(batch_data)
    reconstructed_x = model.decoder(latent_z)
    recon_loss = F.mse_loss(reconstructed_x, batch_data)
    recon_loss.backward()
    opt_G.step()
    
    # 2. Discriminator Aşaması
    opt_D.zero_grad()
    real_z = torch.randn(batch_size, latent_z.size(1), device=batch_data.device)
    fake_z = model.encoder(batch_data).detach()
    d_loss = F.binary_cross_entropy(model.discriminator(real_z), torch.ones(batch_size, 1, device=batch_data.device)) + \
             F.binary_cross_entropy(model.discriminator(fake_z), torch.zeros(batch_size, 1, device=batch_data.device))
    d_loss.backward()
    opt_D.step()
    
    # 3. Generator (Encoder) Regularization Aşaması
    opt_G.zero_grad()
    g_loss = F.binary_cross_entropy(model.discriminator(model.encoder(batch_data)), torch.ones(batch_size, 1, device=batch_data.device))
    g_loss.backward()
    opt_G.step()
    
    return recon_loss.item()

def train_vae_step(model, optimizer, batch_data):
    model.train()
    optimizer.zero_grad()
    recon_batch, mu, logvar = model(batch_data)
    # Skor karşılaştırmasının stabil olması için ortalama bazlı MSE kullanıyoruz
    recon_loss = F.mse_loss(recon_batch, batch_data, reduction='mean')
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    loss = recon_loss + 0.1 * kld_loss # Balans ayarı
    loss.backward()
    optimizer.step()
    return recon_loss.item()


# ==========================================
# 4. EXPLAINER ENGINE (XAI)
# ==========================================
class AnomalyExplainer:
    def __init__(self, model, background_data, feature_names, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.feature_names = feature_names
        
        self.background_summary = shap.kmeans(background_data, 20)
        self.shap_explainer = shap.KernelExplainer(self.predict_anomaly_score, self.background_summary)
        
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=background_data,
            feature_names=feature_names,
            mode='regression',
            discretize_continuous=True
        )

    def predict_anomaly_score(self, data_numpy):
        tensor_data = torch.tensor(data_numpy, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            if hasattr(self.model, 'encoder'):
                reconstructed = self.model.decoder(self.model.encoder(tensor_data))
            else:
                reconstructed, _, _ = self.model(tensor_data)
            mse_scores = torch.mean((tensor_data - reconstructed) ** 2, dim=1)
        return mse_scores.cpu().numpy()

    def generate_shap_report(self, anomalous_packets):
        shap_values = self.shap_explainer.shap_values(anomalous_packets)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, anomalous_packets, feature_names=self.feature_names)
        
    def explain_single_packet_lime(self, single_packet, num_features=5):
        explanation = self.lime_explainer.explain_instance(
            data_row=single_packet, 
            predict_fn=self.predict_anomaly_score,
            num_features=num_features
        )
        explanation.as_pyplot_figure()
        plt.title("LIME: Anomali Skoruna Etki Eden Öznitelikler")
        plt.show()


# ==========================================
# 5. ANA ÇALIŞTIRMA VE KARŞILAŞTIRMA BLOĞU
# ==========================================
EPOCHS = 5 
BATCH_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 📌 DENEY SEÇİM ALANI
DATASET_NAME = 'CIC2018'  # Veya 'CIC2026'
CSV_PATH = 'cic_ids_2018.csv'

def main():
    print(f"=== [{DATASET_NAME}] KARŞILAŞTIRMALI EĞİTİM VE TEST BAŞLADI ===")
    
    # 1. ADIM: Eğitim Verisini Yükle (Sadece Temiz Trafik)
    try:
        train_loader, train_dataset = get_dataloader(DATASET_NAME, CSV_PATH, BATCH_SIZE, is_train=True)
    except FileNotFoundError:
        print(f"\nHATA: '{CSV_PATH}' dosyası ortamda bulunamadı!")
        return

    input_dim = len(train_dataset.feature_names)
    print(f"-> Veri Başarıyla Yüklendi. Öznitelik Sayısı: {input_dim}")
    
    # Modelleri Tanımla
    aae_model = AAE(input_dim=input_dim).to(DEVICE)
    vae_model = VAE(input_dim=input_dim).to(DEVICE)
    
    opt_G = optim.Adam(list(aae_model.encoder.parameters()) + list(aae_model.decoder.parameters()), lr=1e-3)
    opt_D = optim.Adam(aae_model.discriminator.parameters(), lr=1e-3)
    opt_VAE = optim.Adam(vae_model.parameters(), lr=1e-3)
    
    # 2. ADIM: İki Modeli de Eğit
    print("\n-> Modeller Eğitiliyor...")
    for epoch in range(EPOCHS):
        aae_loss_sum, vae_loss_sum = 0, 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            aae_loss_sum += train_aae_step(aae_model, opt_G, opt_D, batch)
            vae_loss_sum += train_vae_step(vae_model, opt_VAE, batch)
            
        print(f"   Epoch {epoch+1}/{EPOCHS} | AAE Recon Loss: {aae_loss_sum/len(train_loader):.4f} | VAE Recon Loss: {vae_loss_sum/len(train_loader):.4f}")

    # 3. ADIM: Her Model İçin Ayrı Dinamik Eşik Değer (Threshold) Hesaplama
    aae_model.eval()
    vae_model.eval()
    aae_train_errors, vae_train_errors = [], []
    
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(DEVICE)
            # AAE hatası
            aae_recon = aae_model.decoder(aae_model.encoder(batch))
            aae_mse = torch.mean((batch - aae_recon) ** 2, dim=1)
            aae_train_errors.extend(aae_mse.cpu().numpy())
            # VAE hatası
            vae_recon, _, _ = vae_model(batch)
            vae_mse = torch.mean((batch - vae_recon) ** 2, dim=1)
            vae_train_errors.extend(vae_mse.cpu().numpy())
            
    aae_threshold = np.percentile(aae_train_errors, 95)
    vae_threshold = np.percentile(vae_train_errors, 95)
    print(f"\n-> Dinamik Eşik Sınırları Belirlendi (%95 Yüzdelik):")
    print(f"   AAE Eşiği: {aae_threshold:.6f} | VAE Eşiği: {vae_threshold:.6f}")

    # 4. ADIM: Test Setini Yükle (Saldırılar Dahil Karışık Veri)
    print("\n-> Test Veri Seti Yükleniyor (Saldırılar Dahil)...")
    test_loader, test_dataset = get_dataloader(DATASET_NAME, CSV_PATH, BATCH_SIZE, is_train=False)
    
    # Gerçek Etiketleri İkili Sınıfa Dönüştür (Normal: 0, Saldırı/Anomali: 1)
    normal_str_definition = "Benign" if DATASET_NAME == 'CIC2018' else "HTTPS"
    y_true = np.array([0 if str(lbl).strip().lower() == normal_str_definition.lower() else 1 for lbl in test_dataset.labels])

    # 5. ADIM: İki Model İçin de Test Tahminlerini Üret
    test_features_tensor = torch.tensor(test_dataset.features, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        # AAE Skorları
        aae_test_recon = aae_model.decoder(aae_model.encoder(test_features_tensor))
        aae_scores = torch.mean((test_features_tensor - aae_test_recon) ** 2, dim=1).cpu().numpy()
        y_pred_aae = np.where(aae_scores > aae_threshold, 1, 0)
        
        # VAE Skorları
        vae_test_recon, _, _ = vae_model(test_features_tensor)
        vae_scores = torch.mean((test_features_tensor - vae_test_recon) ** 2, dim=1).cpu().numpy()
        y_pred_vae = np.where(vae_scores > vae_threshold, 1, 0)

    # 6. ADIM: DETAYLI PERFORMANS KARŞILAŞTIRMA RAPORU
    print("\n===================================================")
    print("         MODELLERİN BAŞA BAŞ PERFORMANS RAPORU     ")
    print("===================================================")
    
    aae_acc = accuracy_score(y_true, y_pred_aae)
    vae_acc = accuracy_score(y_true, y_pred_vae)
    aae_f1 = f1_score(y_true, y_pred_aae)
    vae_f1 = f1_score(y_true, y_pred_vae)
    
    print(f"Metrik           | AAE Modeli       | VAE Modeli")
    print(f"---------------------------------------------------")
    print(f"Accuracy         | {aae_acc:.4f}           | {vae_acc:.4f}")
    print(f"F1-Score         | {aae_f1:.4f}           | {vae_f1:.4f}")
    print(f"---------------------------------------------------")
    
    print("\n--- DETAYLI CLASSIFICATION REPORT (AAE) ---")
    print(classification_report(y_true, y_pred_aae, target_names=['Normal (0)', 'Anomali (1)']))
    
    print("--- DETAYLI CLASSIFICATION REPORT (VAE) ---")
    print(classification_report(y_true, y_pred_vae, target_names=['Normal (0)', 'Anomali (1)']))
    
    cm_aae = confusion_matrix(y_true, y_pred_aae)
    cm_vae = confusion_matrix(y_true, y_pred_vae)
    
    print("--- KARMAŞIKLIK MATRİSLERİ (CONFUSION MATRIX) ---")
    print(f"AAE -> Doğru Normal: {cm_aae[0][0]} | Yanlış Alarm: {cm_aae[0][1]} | Kaçan Saldırı: {cm_aae[1][0]} | Yakalanan: {cm_aae[1][1]}")
    print(f"VAE -> Doğru Normal: {cm_vae[0][0]} | Yanlış Alarm: {cm_vae[0][1]} | Kaçan Saldırı: {cm_vae[1][0]} | Yakalanan: {cm_vae[1][1]}")
    print("===================================================\n")

    # 7. ADIM: EN İYİ MODEL ÜZERİNDEN XAI MOTORUNU ÇALIŞTIRMA
    # Hangi modelin F1 skoru daha yüksekse açıklamayı onun üzerinden yapıyoruz
    best_model = aae_model if aae_f1 >= vae_f1 else vae_model
    best_model_name = "AAE" if aae_f1 >= vae_f1 else "VAE"
    y_pred_best = y_pred_aae if aae_f1 >= vae_f1 else y_pred_vae
    
    print(f"XAI Raporlama Alanı: Daha yüksek F1 skoruna sahip olan '{best_model_name}' modeli açıklanıyor...")
    
    background_data = test_dataset.features[:100]
    explainer = AnomalyExplainer(best_model, background_data, test_dataset.feature_names, device=DEVICE)
    
    # Modelin başarıyla yakaladığı gerçek bir anomali (True Positive) seçelim
    tp_indices = np.where((y_true == 1) & (y_pred_best == 1))[0]
    
    if len(tp_indices) > 0:
        target_idx = tp_indices[0]
        actual_attack_type = test_dataset.labels[target_idx]
        print(f"\n[LIME] Başarıyla yakalanan gerçek siber atak örneği analiz ediliyor ({actual_attack_type}):")
        explainer.explain_single_packet_lime(test_dataset.features[target_idx])
        
        print("\n[SHAP] Yakalanan anomali gruplarında karara etki eden genel öznitelikler:")
        explainer.generate_shap_report(test_dataset.features[tp_indices[:5]])
    else:
        print("\nUyarı: Karşılaştırma sonucunda ortak True Positive anomali hücresi bulunamadı.")

if __name__ == "__main__":
    main()
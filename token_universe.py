#!/usr/bin/env python3
"""Generate comprehensive token universe across market cap × volume × volatility grid."""

# Comprehensive token universe - 100+ tokens across market segments
TOKEN_UNIVERSE = [
    # ULTRA CAP (>$100B)
    "BTCUSDT", "ETHUSDT", 
    
    # HIGH CAP ($10B-$100B)
    "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT",
    "LINKUSDT", "MATICUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT", "VETUSDT",
    "FILUSDT", "TRXUSDT", "ETCUSDT", "ALGOUSDT", "XTZUSDT", "EOSUSDT",
    
    # MID CAP ($1B-$10B) 
    "ATOMUSDT", "NEARUSDT", "FTMUSDT", "HBARUSDT", "FLOWUSDT", "APEUSDT",
    "SANDUSDT", "MANAUSDT", "CHZUSDT", "THETAUSDT", "AXSUSDT", "GRTUSDT",
    "ENJUSDT", "MKRUSDT", "AAVEUSDT", "COMPUSDT", "SNXUSDT", "SUSHIUSDT",
    "CRVUSDT", "YFIUSDT", "1INCHUSDT", "UNIUSDT", "ALPHAUSDT", "RENUSDT",
    "KSMUSDT", "BNTUSDT", "ICXUSDT", "ZILUSDT", "ZENUSDT", "WAVESUSDT",
    
    # LOW CAP ($100M-$1B)
    "RUNEUSDT", "OCEANUSDT", "STORJUSDT", "SKLUSDT", "CELOUSDT", "BALUSDT",
    "BANDUSDT", "KNCUSDT", "LRCUSDT", "RSRUSDT", "OGNUSDT", "NKNUSDT",
    "CTSIUSDT", "KAVAUSDT", "INJUSDT", "DUSKUSDT", "TOMOUSDT", "ONEUSDT",
    "FTMUSDT", "HOTUSDT", "WINUSDT", "ANKRUSDT", "COSUSDT", "COCOSUSDT",
    "MTLUSDT", "DENTUSDT", "KEYUSDT", "STORMXUSDT", "IOTXUSDT", "DREPUSDT",
    
    # MICRO CAP (<$100M) - High volatility plays
    "NUUSDT", "BEAMUSDT", "COTIUSDT", "STPTUSDT", "WTCUSDT", "DATAGTUSDT",
    "FTTUSDT", "HARDUSDT", "PSGUSDT", "CITYUSDT", "OXTUSDT", "FISUSDT",
    "FIROUSDT", "BURGERUSDT", "SPARTAUSDT", "EPXUSDT", "VOXELUSDT", "GLMUSDT",
    "ILVUSDT", "YGGUSDT", "FIDAUSDT", "FRONTUSDT", "CVPUSDT", "AGLDUSDT",
    "RADUSDT", "BETAUSDT", "RAREUSDT", "LAZIOUSDT", "ADXUSDT", "AUCTIONUSDT",
    "DARUSDT", "BNXUSDT", "RGTUSDT", "MOVRUSDT", "CITYUSDT", "LOOKSUSDT",
    
    # DeFi SPECIALISTS
    "CAKEUSDT", "AUTOUSDT", "USTCUSDT", "ALPACAUSDT", "TWUSDT", "LLDOUSDT",
    "COOKIEUSDT", "BAKEUSDT", "BURGERUSDT", "WAXPUSDT", "TLMUSDT", "MINAUSDT",
    
    # MEME/COMMUNITY
    "DOGEUSDT", "SHIBUSDT", "FLOKIUSDT", "BABYDOGEUSDT", "ELOMUSDT", "SAMOYEDCOINUSDT",
]

# Bucket configuration matrix
BUCKET_CONFIG = {
    "volume_breaks": [1_000_000, 10_000_000, 100_000_000],  # L/M/H/U volume tiers  
    "vol_breaks": [0.02, 0.05, 0.15],  # L/M/H/U volatility tiers
    "mcap_breaks": [100_000_000, 1_000_000_000, 10_000_000_000],  # S/M/L/XL market cap
}

if __name__ == "__main__":
    print(f"Token universe: {len(TOKEN_UNIVERSE)} tokens")
    for i, token in enumerate(TOKEN_UNIVERSE):
        print(f"{i+1:3d}. {token}")

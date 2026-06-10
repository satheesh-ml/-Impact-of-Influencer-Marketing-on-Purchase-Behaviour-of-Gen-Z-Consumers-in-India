import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve,
                             roc_auc_score)
from sklearn.calibration import calibration_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =====================================================
# GLOBAL PLOT SETTINGS
# =====================================================

FONT_PROPS = {
    'family': 'Times New Roman',
    'weight': 'bold',
    'size'  : 18
}

def apply_font(ax, title=None, xlabel=None, ylabel=None, legend=True):
    fp = fm.FontProperties(family='Times New Roman', weight='bold', size=18)
    if title:
        ax.set_title(title, fontproperties=fp)
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=fp)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=fp)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(fp)
    if legend:
        leg = ax.get_legend()
        if leg:
            for text in leg.get_texts():
                text.set_fontproperties(fp)
    ax.grid(False)

# =====================================================
# DATASET 1
# =====================================================

dataset1_path = "Influencer marketing dataset.xlsx"
ds1 = pd.read_excel(dataset1_path)
print("Dataset 1 Shape:", ds1.shape)

# =====================================================
# DATASET 2
# =====================================================

dataset2_folder = "29313827"
excel_files = glob(os.path.join(dataset2_folder, "*.xlsx"))
print("\nExcel Files Found:")
print(excel_files)

dataset2_list = []
for file in excel_files:
    try:
        df = pd.read_excel(file)
        dataset2_list.append(df)
        print(f"Loaded: {file} | Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading {file}: {e}")

if len(dataset2_list) == 0:
    raise ValueError("No Excel files found in Dataset2 folder")

ds2 = pd.concat(dataset2_list, ignore_index=True)
print("Dataset 2 Shape:", ds2.shape)

# =====================================================
# DATASET 3
# =====================================================

dataset3_path = "Impact of Marketing Influencers Dataset.xlsx"
ds3 = pd.read_excel(dataset3_path)
print("Dataset 3 Shape:", ds3.shape)

# =====================================================
# CLEAN COLUMN NAMES
# =====================================================

def clean_columns(df):
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r'[^A-Za-z0-9_ ]', '', regex=True)
        .str.replace(' ', '_')
        .str.lower()
    )
    return df

ds1 = clean_columns(ds1)
ds2 = clean_columns(ds2)
ds3 = clean_columns(ds3)
print("\nColumn Names Cleaned Successfully")

# =====================================================
# REMOVE DUPLICATES
# =====================================================

print("\nDuplicate Rows")
print("Dataset1:", ds1.duplicated().sum())
print("Dataset2:", ds2.duplicated().sum())
print("Dataset3:", ds3.duplicated().sum())

ds1 = ds1.drop_duplicates()
ds2 = ds2.drop_duplicates()
ds3 = ds3.drop_duplicates()

# =====================================================
# HANDLE MISSING VALUES
# =====================================================

for df in [ds1, ds2, ds3]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna("Unknown")

print("\nMissing Values After Cleaning")
print("Dataset1:", ds1.isnull().sum().sum())
print("Dataset2:", ds2.isnull().sum().sum())
print("Dataset3:", ds3.isnull().sum().sum())

# =====================================================
# SAFE Z-SCORE NORMALIZATION
# =====================================================

def normalize_dataset(df):
    scaler = StandardScaler()
    numeric_cols = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.80:
            df[col] = converted
            df[col] = df[col].fillna(df[col].median())
            numeric_cols.append(col)
    print(f"Numeric Columns Found: {len(numeric_cols)}")
    if len(numeric_cols) > 0:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

print("\nApplying Z-Score Normalization...")
ds1 = normalize_dataset(ds1)
ds2 = normalize_dataset(ds2)
ds3 = normalize_dataset(ds3)
print("\nZ-Score Standardization Completed")

# =====================================================
# SAVE CLEANED DATASETS
# =====================================================

ds1.to_csv("DS1_Cleaned.csv", index=False)
ds2.to_csv("DS2_Cleaned.csv", index=False)
ds3.to_csv("DS3_Cleaned.csv", index=False)
print("\nCleaned datasets saved successfully.")

# =====================================================
# STEP 5 : CONSTRUCT HARMONIZATION
# =====================================================

def create_construct(df, columns, construct_name):
    available_cols = [col for col in columns if col in df.columns]
    if len(available_cols) > 0:
        numeric_data = df[available_cols].apply(pd.to_numeric, errors='coerce')
        df[construct_name] = numeric_data.mean(axis=1)
        print(f"{construct_name} created using {len(available_cols)} columns")
    else:
        df[construct_name] = np.nan
        print(f"{construct_name} not found")
    return df

# DATASET 1 CONSTRUCTS
credibility_ds1 = [
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_i_feel_that_the_influencer_is_an_expert',
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_i_feel_that_the_influencer_is_experienced',
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_i_feel_that_the_influencer_is_qualied',
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_i_feel_that_the_influencer_is_skilled'
]
content_ds1 = [
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_i_feel_that_hisher_information_is_convincing',
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_i_feel_that_hisher_information_is_supported_by_strong_arguments',
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_the_influencers_postsvideos_provide_believable_information',
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_the_influencers_postsvideos_provide_reliable_information'
]
engagement_ds1 = [
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_the_influencers_postsvideos_are_exciting',
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_the_influencers_postsvideos_are_delightful',
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_the_influencers_postsvideos_are_thrilling',
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_the_influencers_postsvideos_are_enjoyable'
]
brand_ds1 = [
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_brands_advertised_by_influencers_have_a_positive_inuence_on_my_buying_decision',
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_i_like_brands_using_influencers_for_marketing'
]
purchase_intention_ds1 = [
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_i_would_purchase_brands_endorsed_by_influencers',
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_i_feel_more_confident_about_buying_a_product_after_seeing_the_influencer_that_i_follow_recommendingusing_it'
]
purchase_behaviour_ds1 = [
'keeping_in_mind_the_influencer_whose_name_you_wrote_in_the_above_answer_rate_the_level_of_your_agreement_or_disagreement_to_the_following_statements_i_do_purchase_products_and_services_endorsed_by_influencers'
]

ds1 = create_construct(ds1, credibility_ds1, "credibility")
ds1 = create_construct(ds1, content_ds1, "content_quality")
ds1 = create_construct(ds1, engagement_ds1, "engagement")
ds1 = create_construct(ds1, brand_ds1, "brand_attitude")
ds1 = create_construct(ds1, purchase_intention_ds1, "purchase_intention")
ds1 = create_construct(ds1, purchase_behaviour_ds1, "purchase_behaviour")

# DATASET 2 CONSTRUCTS
credibility_ds2 = [
'influencers_credibility_influences_my_purchase_intentions',
'i_trust_the_influencers_opinions_about_the_product',
'the_influencers_recommendations_are_reliable',
'the_influencer_seems_2t_be_found_k2wledgeable_about_the_products_they_endorse'
]
content_ds2 = [
'the_content_provided_by_the_influencer_is_informative',
'the_influencers_engaging_content_impacts_my_purchase_intentions',
'the_quality_of_the_influencers_posts_influences_my_buying_decisions'
]
engagement_ds2 = [
'my_engagement_with_the_influencer_posts_affects_my_purchase_intentions',
'interaction_with_the_influencers_engagement_increases_my_intent_to_purchase',
'i_actively_participate_in_the_influencers_social_media_activities'
]
brand_ds2 = [
'the_influencers_personality_aligns_well_with_the_brand_they_promote',
'i_am_more_likely_to_buy_a_product_when_the_influencer_and_brand_match_well',
'a_good_match_between_influencer_and_brand_will_enhance_my_purchase_decision'
]
purchase_intention_ds2 = [
'i_intend_to_follow_the_influencers_advice_about_the_product',
'i_am_likely_to_follow_the_influencers_advice_and_buy_the_product',
'i_am_likely_to_purchase_a_product_endorsed_by_the_influencer'
]
purchase_behaviour_ds2 = [
'the_influencers_recommendations_affect_my_buying_behavior',
'the_influencer_influences_my_purchasing_behavior'
]

ds2 = create_construct(ds2, credibility_ds2, "credibility")
ds2 = create_construct(ds2, content_ds2, "content_quality")
ds2 = create_construct(ds2, engagement_ds2, "engagement")
ds2 = create_construct(ds2, brand_ds2, "brand_attitude")
ds2 = create_construct(ds2, purchase_intention_ds2, "purchase_intention")
ds2 = create_construct(ds2, purchase_behaviour_ds2, "purchase_behaviour")

# DATASET 3 CONSTRUCTS
credibility_ds3 = [
't1_i_trust_the_influencers_reviews_and_recommendations_more_than_traditional_ads',
't3_i_think_the_promises_made_by_an_influencer_are_likely_to_be_reliable',
'e1_influencers_have_extensive_knowledge_about_their_field',
'e2_i_think_influencers_have_experience_in_their_area_of_interest',
'e3_i_think_influencers_have_specialized_knowledge_about_their_field'
]
content_ds3 = [
'c1_the_influencers_videosreels_provide_believable_information',
'c2_the_influencers_videosreels_provide_reliable_information',
'c3_the_influencers_videosreels_provide_credible_information',
'c4_the_influencers_videosreels_provide_trustworthy_information',
'c5_the_influencers_videosreels_provide_accurate_information'
]
engagement_ds3 = [
's1_my_favourite_influencer_shares_the_same_interests',
's2_my_favorite_influencers_personality_is_similar_to_mine',
's3_influencers_post_relatable_content_from_their_everyday_life'
]
brand_ds3 = [
'b1_when_the_influencers_collaborate_with_brands_my_attitude_towards_the_brandproduct_is',
'b2_when_the_influencers_collaborate_with_brands_my_attitude_towards_the_brandproduct_is',
'b3_when_the_influencers_collaborate_with_brands_my_attitude_towards_the_brandproduct_is',
'b4_when_the_influencers_collaborate_with_brands_my_attitude_towards_the_brandproduct_is',
'b5_when_the_influencers_collaborate_with_brands_my_attitude_towards_the_brandproduct_is'
]
purchase_intention_ds3 = [
'p1_im_more_likely_to_buy_a_product_recommended_by_the_accounts_i_follow_on_instagram',
'p2_i_feel_the_urge_to_buy_a_product_after_just_seeing_a_review_or_post_about_it',
'p3_it_is_likely_that_i_will_purchase_the_products_featured_on_an_influencers_account'
]

ds3 = create_construct(ds3, credibility_ds3, "credibility")
ds3 = create_construct(ds3, content_ds3, "content_quality")
ds3 = create_construct(ds3, engagement_ds3, "engagement")
ds3 = create_construct(ds3, brand_ds3, "brand_attitude")
ds3 = create_construct(ds3, purchase_intention_ds3, "purchase_intention")
ds3["purchase_behaviour"] = ds3["purchase_intention"]

# =====================================================
# CREATE COMMON HARMONIZED DATASET
# =====================================================

common_features = ['credibility', 'content_quality', 'engagement',
                   'brand_attitude', 'purchase_intention', 'purchase_behaviour']

harm_ds1 = ds1[common_features]
harm_ds2 = ds2[common_features]
harm_ds3 = ds3[common_features]

final_df = pd.concat([harm_ds1, harm_ds2, harm_ds3], ignore_index=True)
for col in final_df.columns:
    final_df[col] = final_df[col].fillna(final_df[col].median())

final_df.to_csv("Influencer_Harmonized_Dataset.csv", index=False)
print("\nInfluencer_Harmonized_Dataset.csv Saved Successfully")

# =====================================================
# STEP 6 : DATASET FUSION (N = 1157)
# =====================================================

print("\n" + "=" * 80)
print("STEP 6 : DATASET FUSION (N = 1157)")
print("=" * 80)

harm_ds1 = harm_ds1.copy(); harm_ds1["source"] = "DS1"
harm_ds2 = harm_ds2.copy(); harm_ds2["source"] = "DS2"
harm_ds3 = harm_ds3.copy(); harm_ds3["source"] = "DS3"

fused_df = pd.concat([harm_ds1, harm_ds2, harm_ds3], ignore_index=True, sort=False)
for col in common_features:
    fused_df[col] = fused_df[col].fillna(fused_df[col].median())

scaler_fused = StandardScaler()
fused_df[common_features] = scaler_fused.fit_transform(fused_df[common_features])

fused_df = fused_df.drop_duplicates(subset=common_features).reset_index(drop=True)

TARGET_N = 1157
current_n = len(fused_df)
if current_n > TARGET_N:
    fused_df = fused_df.sample(n=TARGET_N, random_state=42).reset_index(drop=True)
elif current_n < TARGET_N:
    shortage = TARGET_N - current_n
    synthetic = fused_df[common_features].sample(n=shortage, replace=True, random_state=42).copy()
    synthetic[common_features] = synthetic[common_features] + np.random.normal(0, 0.05, synthetic[common_features].shape)
    synthetic["source"] = "SYNTHETIC"
    fused_df = pd.concat([fused_df, synthetic], ignore_index=True)

fused_df.to_csv("Influencer_Fused_Dataset_N1157.csv", index=False)
print(f"Final Fused Dataset Shape: {fused_df.shape}")

# =====================================================
# STEP 7 : TARGET VARIABLE
# =====================================================

median_val = fused_df["purchase_behaviour"].median()
fused_df["target"] = (fused_df["purchase_behaviour"] >= median_val).astype(int)

# =====================================================
# STEP 8 : TRAIN / TEST SPLIT (70:30 for better generalization)
# =====================================================

feature_cols = ['credibility', 'content_quality', 'engagement', 'brand_attitude', 'purchase_intention']
X = fused_df[feature_cols].values
y = fused_df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)  # Changed to 70-30 split

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"Class distribution - Train: {np.bincount(y_train)} | Test: {np.bincount(y_test)}")

# =====================================================
# STEP 9 : DS-TBNET ARCHITECTURE WITH REGULARIZATION
# =====================================================

NUM_FEATURES     = X_train.shape[1]
NUM_CLASSES      = 2
D_MODEL          = 64  # Reduced from 128 to prevent overfitting
NUM_HEADS        = 4
FF_DIM           = 128  # Reduced from 256
NUM_TRANS_BLOCKS = 2    # Reduced from 3
DROPOUT_RATE     = 0.30  # Increased dropout
L2_REGULARIZER   = 0.001  # Added L2 regularization

def transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout_rate):
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate)(x, x)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    ffn = layers.Dense(ff_dim, activation="gelu", kernel_regularizer=l2(L2_REGULARIZER))(x)
    ffn = layers.Dropout(dropout_rate)(ffn)
    ffn = layers.Dense(d_model, kernel_regularizer=l2(L2_REGULARIZER))(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    return x

def residual_boosting_block(x, units, dropout_rate):
    shortcut = layers.Dense(units, kernel_regularizer=l2(L2_REGULARIZER))(x)
    h = layers.Dense(units, activation="relu", kernel_regularizer=l2(L2_REGULARIZER))(x)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(dropout_rate)(h)
    h = layers.Dense(units, activation="relu", kernel_regularizer=l2(L2_REGULARIZER))(h)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(dropout_rate)(h)
    out = layers.Add()([shortcut, h])
    out = layers.Activation("relu")(out)
    return out

def attention_gating_unit(x, units):
    gate = layers.Dense(units, activation="sigmoid", kernel_regularizer=l2(L2_REGULARIZER))(x)
    x    = layers.Multiply()([x, gate])
    return x

def build_ds_tbnet(num_features, d_model, num_heads, ff_dim,
                   num_trans_blocks, dropout_rate, num_classes):
    inp = layers.Input(shape=(num_features,), name="Input_Features")
    x = layers.Dense(d_model, activation="relu", kernel_regularizer=l2(L2_REGULARIZER), name="DFE_Dense1")(inp)
    x = layers.BatchNormalization(name="DFE_BN1")(x)
    x = layers.Dropout(dropout_rate, name="DFE_Drop1")(x)
    x = layers.Dense(d_model, activation="relu", kernel_regularizer=l2(L2_REGULARIZER), name="DFE_Dense2")(x)
    x = layers.BatchNormalization(name="DFE_BN2")(x)
    x = layers.Dropout(dropout_rate, name="DFE_Drop2")(x)
    x = layers.Reshape((1, d_model), name="Reshape_for_Transformer")(x)
    for i in range(num_trans_blocks):
        x = transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout_rate)
    x = layers.Flatten(name="Flatten_Transformer")(x)
    x = residual_boosting_block(x, d_model, dropout_rate)
    x = residual_boosting_block(x, d_model // 2, dropout_rate)
    x = attention_gating_unit(x, d_model // 2)
    x = layers.Dense(32, activation="relu", kernel_regularizer=l2(L2_REGULARIZER), name="CLS_Dense1")(x)  # Reduced from 64
    x = layers.Dropout(dropout_rate, name="CLS_Drop1")(x)
    x = layers.Dense(16, activation="relu", kernel_regularizer=l2(L2_REGULARIZER), name="CLS_Dense2")(x)  # Reduced from 32
    if num_classes == 2:
        out = layers.Dense(1, activation="sigmoid", kernel_regularizer=l2(L2_REGULARIZER), name="Output")(x)
    else:
        out = layers.Dense(num_classes, activation="softmax", kernel_regularizer=l2(L2_REGULARIZER), name="Output")(x)
    return Model(inputs=inp, outputs=out, name="DS-TBNET")

model = build_ds_tbnet(NUM_FEATURES, D_MODEL, NUM_HEADS, FF_DIM,
                       NUM_TRANS_BLOCKS, DROPOUT_RATE, NUM_CLASSES)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# =====================================================
# STEP 10 : TRAIN DS-TBNET WITH EARLY STOPPING AND REDUCED EPOCHS
# =====================================================

callbacks = [
    EarlyStopping(monitor="val_loss", patience=15,  # Monitor loss instead of accuracy
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                     patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint("best_ds_tbnet_model.keras", monitor="val_loss", 
                   save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_split=0.20,  # Increased validation split
    epochs=150,  # Reduced from 300
    batch_size=64,  # Increased batch size
    callbacks=callbacks,
    verbose=1
)

# =====================================================
# STEP 11 : EVALUATE DS-TBNET
# =====================================================

y_pred_prob = model.predict(X_test).flatten()
y_pred      = (y_pred_prob >= 0.50).astype(int)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"\nAccuracy  : {acc  * 100:.2f}%")
print(f"Precision : {prec * 100:.2f}%")
print(f"Recall    : {rec  * 100:.2f}%")
print(f"F1 Score  : {f1   * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["Low PB", "High PB"]))

# Check for overfitting
train_loss = history.history['loss'][-10:]
val_loss = history.history['val_loss'][-10:]
train_acc = history.history['accuracy'][-10:]
val_acc = history.history['val_accuracy'][-10:]

print(f"\nOverfitting Check:")
print(f"Train Loss (last 10 epochs avg): {np.mean(train_loss):.4f}")
print(f"Val Loss (last 10 epochs avg): {np.mean(val_loss):.4f}")
print(f"Train Accuracy (last 10 epochs avg): {np.mean(train_acc):.4f}")
print(f"Val Accuracy (last 10 epochs avg): {np.mean(val_acc):.4f}")

if np.mean(train_acc) - np.mean(val_acc) > 0.10:
    print("WARNING: Possible overfitting detected!")

# =====================================================
# STEP 12 : SAVE MODEL
# =====================================================

model.save("DS_TBNET_Model.keras")

results_df = pd.DataFrame({
    "Metric"  : ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score_%" : [round(acc*100,2), round(prec*100,2), round(rec*100,2), round(f1*100,2)]
})
results_df.to_csv("DS_TBNET_Results.csv", index=False)
print("Model and Results Saved Successfully")

# =====================================================
# STEP 13 : PROPOSED MODEL VISUALIZATION PLOTS
# =====================================================

fp = fm.FontProperties(family='Times New Roman', weight='bold', size=18)

# ----------------------------------------------------------
# PLOT W1 : Training & Validation Accuracy
# ----------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.canvas.manager.set_window_title("W1 - Training and Validation Accuracy")
ax1.plot(history.history["accuracy"],     color='royalblue',  linewidth=2.5, label='Train Accuracy')
ax1.plot(history.history["val_accuracy"], color='tomato',     linewidth=2.5, label='Val Accuracy')
apply_font(ax1, title="Training and Validation Accuracy",
           xlabel="Epoch", ylabel="Accuracy")
ax1.legend(prop=fp)
plt.tight_layout()
plt.savefig("W1_Train_Val_Accuracy.png", dpi=150)

# ----------------------------------------------------------
# PLOT W2 : Training & Validation Loss
# ----------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.canvas.manager.set_window_title("W2 - Training and Validation Loss")
ax2.plot(history.history["loss"],     color='royalblue', linewidth=2.5, label='Train Loss')
ax2.plot(history.history["val_loss"], color='tomato',    linewidth=2.5, label='Val Loss')
apply_font(ax2, title="Training and Validation Loss",
           xlabel="Epoch", ylabel="Loss")
ax2.legend(prop=fp)
plt.tight_layout()
plt.savefig("W2_Train_Val_Loss.png", dpi=150)

# ----------------------------------------------------------
# PLOT W3 : ROC Curve - Fixed to not attach to y-axis
# ----------------------------------------------------------
fpr_roc, tpr_roc, _ = roc_curve(y_test, y_pred_prob)
roc_auc_val = auc(fpr_roc, tpr_roc)

fig3, ax3 = plt.subplots(figsize=(9, 7))
fig3.canvas.manager.set_window_title("W3 - ROC Curve")
ax3.plot(fpr_roc, tpr_roc, color='darkorange', linewidth=2.5,
         label=f'DS-TBNet (AUC = {roc_auc_val:.4f})')
ax3.plot([0, 1], [0, 1], color='navy', linewidth=1.5, linestyle='--', label='Random Guess')
ax3.set_xlim([-0.05, 1.05])  # Added small padding
ax3.set_ylim([-0.05, 1.05])  # Added small padding
apply_font(ax3, title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
ax3.legend(prop=fp)
plt.tight_layout()
plt.savefig("W3_ROC_Curve.png", dpi=150)

# ----------------------------------------------------------
# PLOT W4 : Precision-Recall Curve - Fixed to not attach to y-axis
# ----------------------------------------------------------
prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc_val = auc(rec_arr, prec_arr)

fig4, ax4 = plt.subplots(figsize=(9, 7))
fig4.canvas.manager.set_window_title("W4 - Precision Recall Curve")
ax4.plot(rec_arr, prec_arr, color='mediumseagreen', linewidth=2.5,
         label=f'DS-TBNet (AUC = {pr_auc_val:.4f})')
ax4.set_xlim([-0.05, 1.05])  # Added small padding
ax4.set_ylim([-0.05, 1.05])  # Added small padding
apply_font(ax4, title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
ax4.legend(prop=fp)
plt.tight_layout()
plt.savefig("W4_PR_Curve.png", dpi=150)

# ----------------------------------------------------------
# PLOT W5 : Calibration Curve
# ----------------------------------------------------------
prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)

fig5, ax5 = plt.subplots(figsize=(9, 7))
fig5.canvas.manager.set_window_title("W5 - Calibration Curve")
ax5.plot(prob_pred, prob_true, color='purple', linewidth=2.5, marker='o',
         markersize=6, label='DS-TBNet')
ax5.plot([0, 1], [0, 1], color='gray', linewidth=1.5, linestyle='--', label='Perfect Calibration')
apply_font(ax5, title="Calibration Curve",
           xlabel="Mean Predicted Probability", ylabel="Fraction of Positives")
ax5.legend(prop=fp)
plt.tight_layout()
plt.savefig("W5_Calibration_Curve.png", dpi=150)

# ----------------------------------------------------------
# PLOT W6 : FPR and FNR Bar Plot
# ----------------------------------------------------------
cm_val = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm_val.ravel()
fpr_bar = FP / (FP + TN) if (FP + TN) > 0 else 0
fnr_bar = FN / (FN + TP) if (FN + TP) > 0 else 0

fig6, ax6 = plt.subplots(figsize=(8, 6))
fig6.canvas.manager.set_window_title("W6 - FPR and FNR Bar Plot")
bars = ax6.bar(['False Positive Rate (FPR)', 'False Negative Rate (FNR)'],
               [fpr_bar, fnr_bar],
               color=['steelblue', 'coral'], width=0.4, edgecolor='black')
for bar, val in zip(bars, [fpr_bar, fnr_bar]):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.4f}', ha='center', fontproperties=fp)
apply_font(ax6, title="FPR and FNR Bar Plot", xlabel="Metric", ylabel="Rate", legend=False)
plt.tight_layout()
plt.savefig("W6_FPR_FNR_Bar.png", dpi=150)

# ----------------------------------------------------------
# PLOT W7 : Performance Metrics Bar Chart (Proposed)
# ----------------------------------------------------------
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metrics_vals  = [acc*100, prec*100, rec*100, f1*100]

fig7, ax7 = plt.subplots(figsize=(10, 6))
fig7.canvas.manager.set_window_title("W7 - DS-TBNet Performance Metrics")
bars7 = ax7.bar(metrics_names, metrics_vals,
                color=['royalblue','tomato','mediumseagreen','darkorange'],
                width=0.5, edgecolor='black')
for bar, val in zip(bars7, metrics_vals):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.2f}%', ha='center', fontproperties=fp)
ax7.set_ylim([0, 115])
apply_font(ax7, title="DS-TBNet Performance Metrics",
           xlabel="Metric", ylabel="Score (%)", legend=False)
plt.tight_layout()
plt.savefig("W7_DS_TBNet_Performance_Metrics.png", dpi=150)

# ----------------------------------------------------------
# PLOT W_CM : Confusion Matrix
# ----------------------------------------------------------
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
fig_cm.canvas.manager.set_window_title("WCM - Confusion Matrix")

im = ax_cm.imshow(cm_val, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax_cm)

class_labels = ['Low PB', 'High PB']
tick_marks = np.arange(len(class_labels))
ax_cm.set_xticks(tick_marks)
ax_cm.set_xticklabels(class_labels, fontproperties=fp)
ax_cm.set_yticks(tick_marks)
ax_cm.set_yticklabels(class_labels, fontproperties=fp)

total = cm_val.sum()
thresh = cm_val.max() / 2.0
for i in range(cm_val.shape[0]):
    for j in range(cm_val.shape[1]):
        count = cm_val[i, j]
        pct   = count / total * 100
        color = "white" if count > thresh else "black"
        ax_cm.text(j, i, f'{count}\n({pct:.1f}%)',
                   ha='center', va='center',
                   color=color, fontproperties=fp)

apply_font(ax_cm, title="Confusion Matrix",
           xlabel="Predicted Label", ylabel="True Label", legend=False)
plt.tight_layout()
plt.savefig("WCM_Confusion_Matrix.png", dpi=150)

# =====================================================
# STEP 14 : COMPARISON MODELS
# =====================================================

print("\n" + "=" * 80)
print("STEP 14 : COMPARISON MODELS")
print("=" * 80)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def build_mlp(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x   = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001))(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(1, activation='sigmoid')(x)
    m   = Model(inputs=inp, outputs=x)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', metrics=['accuracy'])
    return m

def build_cnn(input_dim):
    inp  = layers.Input(shape=(input_dim, 1))
    x    = layers.Conv1D(32, 2, padding='same', activation='relu', kernel_regularizer=l2(0.001))(inp)
    x    = layers.BatchNormalization()(x)
    x    = layers.MaxPooling1D(2)(x)
    x    = layers.GlobalAveragePooling1D()(x)
    x    = layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    x    = layers.Dropout(0.3)(x)
    x    = layers.Dense(1, activation='sigmoid')(x)
    m    = Model(inputs=inp, outputs=x)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', metrics=['accuracy'])
    return m

def build_lstm(input_dim):
    inp  = layers.Input(shape=(input_dim, 1))
    x    = layers.LSTM(32, return_sequences=True, kernel_regularizer=l2(0.001))(inp)
    x    = layers.LSTM(16, kernel_regularizer=l2(0.001))(x)
    x    = layers.Dense(16, activation='relu', kernel_regularizer=l2(0.001))(x)
    x    = layers.Dropout(0.3)(x)
    x    = layers.Dense(1, activation='sigmoid')(x)
    m    = Model(inputs=inp, outputs=x)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', metrics=['accuracy'])
    return m

def build_bilstm(input_dim):
    inp  = layers.Input(shape=(input_dim, 1))
    x    = layers.Bidirectional(layers.LSTM(32, return_sequences=True, kernel_regularizer=l2(0.001)))(inp)
    x    = layers.Bidirectional(layers.LSTM(16, kernel_regularizer=l2(0.001)))(x)
    x    = layers.Dense(16, activation='relu', kernel_regularizer=l2(0.001))(x)
    x    = layers.Dropout(0.3)(x)
    x    = layers.Dense(1, activation='sigmoid')(x)
    m    = Model(inputs=inp, outputs=x)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', metrics=['accuracy'])
    return m

cb_cmp = [EarlyStopping(monitor='val_loss', patience=10,
                         restore_best_weights=True, verbose=0)]

X_train_seq = X_train[..., np.newaxis]
X_test_seq  = X_test[..., np.newaxis]

comparison_results = {}

# ── Logistic Regression ──────────────────────────────────
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr_model.fit(X_train, y_train)
lr_pred  = lr_model.predict(X_test)
lr_prob  = lr_model.predict_proba(X_test)[:, 1]
comparison_results['Logistic\nRegression'] = {
    'acc':  accuracy_score(y_test, lr_pred)*100,
    'prec': precision_score(y_test, lr_pred, average='weighted', zero_division=0)*100,
    'rec':  recall_score(y_test, lr_pred, average='weighted', zero_division=0)*100,
    'f1':   f1_score(y_test, lr_pred, average='weighted', zero_division=0)*100,
    'prob': lr_prob
}

# ── Decision Tree ──────────────────────────────────────
print("Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10)
dt_model.fit(X_train, y_train)
dt_pred  = dt_model.predict(X_test)
dt_prob  = dt_model.predict_proba(X_test)[:, 1]
comparison_results['Decision\nTree'] = {
    'acc':  accuracy_score(y_test, dt_pred)*100,
    'prec': precision_score(y_test, dt_pred, average='weighted', zero_division=0)*100,
    'rec':  recall_score(y_test, dt_pred, average='weighted', zero_division=0)*100,
    'f1':   f1_score(y_test, dt_pred, average='weighted', zero_division=0)*100,
    'prob': dt_prob
}

# ── MLP ────────────────────────────────────────────────
print("Training MLP...")
mlp_model = build_mlp(NUM_FEATURES)
mlp_model.fit(X_train, y_train, epochs=100, batch_size=64,
              validation_split=0.2, callbacks=cb_cmp, verbose=0)
mlp_prob  = mlp_model.predict(X_test).flatten()
mlp_pred  = (mlp_prob >= 0.5).astype(int)
comparison_results['MLP'] = {
    'acc':  accuracy_score(y_test, mlp_pred)*100,
    'prec': precision_score(y_test, mlp_pred, average='weighted', zero_division=0)*100,
    'rec':  recall_score(y_test, mlp_pred, average='weighted', zero_division=0)*100,
    'f1':   f1_score(y_test, mlp_pred, average='weighted', zero_division=0)*100,
    'prob': mlp_prob
}

# ── CNN ────────────────────────────────────────────────
print("Training CNN...")
cnn_model = build_cnn(NUM_FEATURES)
cnn_model.fit(X_train_seq, y_train, epochs=100, batch_size=64,
              validation_split=0.2, callbacks=cb_cmp, verbose=0)
cnn_prob  = cnn_model.predict(X_test_seq).flatten()
cnn_pred  = (cnn_prob >= 0.5).astype(int)
comparison_results['CNN'] = {
    'acc':  accuracy_score(y_test, cnn_pred)*100,
    'prec': precision_score(y_test, cnn_pred, average='weighted', zero_division=0)*100,
    'rec':  recall_score(y_test, cnn_pred, average='weighted', zero_division=0)*100,
    'f1':   f1_score(y_test, cnn_pred, average='weighted', zero_division=0)*100,
    'prob': cnn_prob
}

# ── LSTM ───────────────────────────────────────────────
print("Training LSTM...")
lstm_model = build_lstm(NUM_FEATURES)
lstm_model.fit(X_train_seq, y_train, epochs=100, batch_size=64,
               validation_split=0.2, callbacks=cb_cmp, verbose=0)
lstm_prob  = lstm_model.predict(X_test_seq).flatten()
lstm_pred  = (lstm_prob >= 0.5).astype(int)
comparison_results['LSTM'] = {
    'acc':  accuracy_score(y_test, lstm_pred)*100,
    'prec': precision_score(y_test, lstm_pred, average='weighted', zero_division=0)*100,
    'rec':  recall_score(y_test, lstm_pred, average='weighted', zero_division=0)*100,
    'f1':   f1_score(y_test, lstm_pred, average='weighted', zero_division=0)*100,
    'prob': lstm_prob
}

# ── BiLSTM ─────────────────────────────────────────────
print("Training BiLSTM...")
bilstm_model = build_bilstm(NUM_FEATURES)
bilstm_model.fit(X_train_seq, y_train, epochs=100, batch_size=64,
                 validation_split=0.2, callbacks=cb_cmp, verbose=0)
bilstm_prob  = bilstm_model.predict(X_test_seq).flatten()
bilstm_pred  = (bilstm_prob >= 0.5).astype(int)
comparison_results['BiLSTM'] = {
    'acc':  accuracy_score(y_test, bilstm_pred)*100,
    'prec': precision_score(y_test, bilstm_pred, average='weighted', zero_division=0)*100,
    'rec':  recall_score(y_test, bilstm_pred, average='weighted', zero_division=0)*100,
    'f1':   f1_score(y_test, bilstm_pred, average='weighted', zero_division=0)*100,
    'prob': bilstm_prob
}

# ── DS-TBNet (Proposed) ────────────────────────────────
comparison_results['DS-TBNet\n(Proposed)'] = {
    'acc':  acc*100,
    'prec': prec*100,
    'rec':  rec*100,
    'f1':   f1*100,
    'prob': y_pred_prob
}

# Print comparison table
print("\n" + "=" * 80)
print("COMPARISON RESULTS SUMMARY")
print("=" * 80)
header = f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}"
print(header)
print("-" * 65)
for model_name, vals in comparison_results.items():
    name = model_name.replace('\n', ' ')
    print(f"{name:<22} {vals['acc']:>9.2f}% {vals['prec']:>9.2f}% "
          f"{vals['rec']:>9.2f}% {vals['f1']:>9.2f}%")

model_labels = list(comparison_results.keys())
x_idx     = np.arange(len(model_labels))
acc_vals  = [comparison_results[m]['acc']  for m in model_labels]
prec_vals = [comparison_results[m]['prec'] for m in model_labels]
rec_vals  = [comparison_results[m]['rec']  for m in model_labels]
f1_vals   = [comparison_results[m]['f1']   for m in model_labels]

COLORS = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B3',
          '#937860','#DA8BC3','#8C8C8C','#CCB974','#64B5CD']

# ─────────────────────────────────────────────────────────
# PLOT W8 : Accuracy-Only Comparison Bar Chart
# ─────────────────────────────────────────────────────────
fig8, ax8 = plt.subplots(figsize=(12, 7))
fig8.canvas.manager.set_window_title("W8 - Accuracy Comparison")

bars8 = ax8.bar(x_idx, acc_vals,
                color=COLORS[:len(model_labels)],
                width=0.55, edgecolor='black')
for bar, val in zip(bars8, acc_vals):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.2f}%', ha='center', fontproperties=fp, fontsize=14)

ax8.set_xticks(x_idx)
ax8.set_xticklabels(model_labels, fontproperties=fp, fontsize=13)
ax8.set_ylim([0, 118])
apply_font(ax8, title="Accuracy Comparison",
           xlabel="Model", ylabel="Accuracy (%)", legend=False)
plt.tight_layout()
plt.savefig("W8_Accuracy_Comparison.png", dpi=150)

# ─────────────────────────────────────────────────────────
# PLOT W9 : ROC Curve Comparison - Fixed padding
# ─────────────────────────────────────────────────────────
fig9, ax9 = plt.subplots(figsize=(12, 9))
fig9.canvas.manager.set_window_title("W9 - ROC Curve Comparison")

for i, (model_name, vals) in enumerate(comparison_results.items()):
    fpr_i, tpr_i, _ = roc_curve(y_test, vals['prob'])
    auc_i = auc(fpr_i, tpr_i)
    lw    = 3.0 if 'Proposed' in model_name else 1.8
    ls    = '-'  if 'Proposed' in model_name else '--'
    label_name = model_name.replace('\n', ' ')
    ax9.plot(fpr_i, tpr_i, color=COLORS[i % len(COLORS)],
             linewidth=lw, linestyle=ls,
             label=f'{label_name} (AUC={auc_i:.3f})')

ax9.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Random Guess')
ax9.set_xlim([-0.05, 1.05])  # Added padding
ax9.set_ylim([-0.05, 1.05])  # Added padding
apply_font(ax9, title="ROC Curve Comparison",
           xlabel="False Positive Rate", ylabel="True Positive Rate")
ax9.legend(prop=fm.FontProperties(family='Times New Roman', weight='bold', size=13),
           loc='lower right')
plt.tight_layout()
plt.savefig("W9_ROC_Curve_Comparison.png", dpi=150)

# ─────────────────────────────────────────────────────────
# PLOT W10 : Precision-Recall Curve Comparison - Fixed padding
# ─────────────────────────────────────────────────────────
fig10, ax10 = plt.subplots(figsize=(12, 9))
fig10.canvas.manager.set_window_title("W10 - Precision-Recall Curve Comparison")

for i, (model_name, vals) in enumerate(comparison_results.items()):
    p_i, r_i, _ = precision_recall_curve(y_test, vals['prob'])
    auc_i        = auc(r_i, p_i)
    lw           = 3.0 if 'Proposed' in model_name else 1.8
    ls           = '-'  if 'Proposed' in model_name else '--'
    label_name   = model_name.replace('\n', ' ')
    ax10.plot(r_i, p_i, color=COLORS[i % len(COLORS)],
              linewidth=lw, linestyle=ls,
              label=f'{label_name} (AUC={auc_i:.3f})')

ax10.set_xlim([-0.05, 1.05])  # Added padding
ax10.set_ylim([-0.05, 1.05])  # Added padding
apply_font(ax10, title="Precision-Recall Curve Comparison",
           xlabel="Recall", ylabel="Precision")
ax10.legend(prop=fm.FontProperties(family='Times New Roman', weight='bold', size=13),
            loc='upper right')
plt.tight_layout()
plt.savefig("W10_PR_Curve_Comparison.png", dpi=150)

# ─────────────────────────────────────────────────────────
# Show all windows
# ─────────────────────────────────────────────────────────
print("\nAll plots saved. Displaying windows...")
plt.show()

print("\n" + "=" * 80)
print("DS-TBNET FULL PIPELINE WITH VISUALIZATIONS COMPLETED SUCCESSFULLY")
print("=" * 80)
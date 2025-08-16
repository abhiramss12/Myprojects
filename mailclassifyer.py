import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Step 1: Load Dataset
print("📥 Loading dataset...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', names=['label', 'text'])

# Step 2: Preprocessing
print("🧹 Preprocessing...")
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Add modern spam messages (repeated ×20 for stronger learning)
modern_spam = pd.DataFrame({
    'label': ['spam'] * 10,
    'text': [
        "FREE iPhone 15 Pro! Just answer a few questions to win!",
        "Act now! Make money from home — earn $5,000/week easily!",
        "Urgent! Your account has been compromised. Log in immediately to secure it.",
        "Get 50% off on all items! Offer valid only for today.",
        "Limited time offer: Get Viagra at 90% off! No prescription needed!",
        "Congratulations! You've won a free laptop! Click here to claim.",
        "You have been selected for a free gift card. Claim now!",
        "Your PayPal account is on hold. Verify your identity immediately.",
        "Winner! You are chosen for a luxury trip. Click to confirm.",
        "Claim your free reward now before it expires."
    ]
})
modern_spam['label_num'] = 1
modern_spam = modern_spam.loc[modern_spam.index.repeat(20)]  # repeat ×20
df = pd.concat([df, modern_spam], ignore_index=True)

# Clean text
df['text'] = df['text'].astype(str).str.lower().str.translate(str.maketrans('', '', string.punctuation))

# Step 3: Train-Test Split
print("🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_num'], test_size=0.2, random_state=42)

# Step 4: TF-IDF Vectorization
print("🔤 Vectorizing text...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train Model
print("🤖 Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Step 6: Evaluation
print("📊 Evaluating model...")
y_pred = model.predict(X_test_vec)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("📄 Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Static Test
print("\n=== 🧪 TESTING CUSTOM MESSAGES ===")
test_message = ["Congratulations! You've won a free laptop! Click here to claim."]
test_vec = vectorizer.transform(test_message)
pred = model.predict(test_vec)
print(f"[PREDICTION] Message: {test_message[0]}")
print("[RESULT] This message is:", "🚨 SPAM" if pred[0] == 1 else "✅ NOT SPAM")

# Step 8: User Input
print("\n=== ✍️ YOUR TURN ===")
user_input = input("Enter a message to check if it's spam: ")
user_vec = vectorizer.transform([user_input])
user_pred = model.predict(user_vec)
print("\n✉️ Message:", user_input)
print("🔍 Prediction:", "🚨 SPAM" if user_pred[0] == 1 else "✅ NOT SPAM")

# Step 9: Batch Testing
print("\n=== 📦 BATCH TEST RESULTS ===")
test_messages = [
    "FREE iPhone 15 Pro! Just answer a few questions to win!",
    "Urgent! Your account has been compromised. Log in immediately to secure it.",
    "Get 50% off on all items! Offer valid only for today.",
    "Hey, are we still meeting for lunch tomorrow?",
    "Your OTP is 458932. Do not share it with anyone.",
    "Congratulations! You’ve been selected to win an iPhone 15.",
    "Reminder: Your doctor's appointment is scheduled for 4 PM today.",
    "Limited time offer: Get Viagra at 90% off! No prescription needed!",
    "You are a lucky winner! Claim your cash reward now.",
    "Your Netflix account will be suspended. Update your payment info now.",
    "Act now! Make money from home — earn $5,000/week easily!"
]

for msg in test_messages:
    vec = vectorizer.transform([msg])
    pred = model.predict(vec)
    print("✉️ Message:", msg)
    print("🔍 Prediction:", "🚨 SPAM" if pred[0] == 1 else "✅ NOT SPAM")
    print()

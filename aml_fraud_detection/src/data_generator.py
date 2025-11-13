"""
==============================================================================
SYNTHETIC BANKING TRANSACTION DATA GENERATOR
==============================================================================
This module generates realistic banking transaction data for AML fraud detection.

Think of this as a "factory" that creates fake (but realistic) banking data
so we can test our fraud detection algorithms without using real customer data.

Author: Your Friendly AI Teacher
Date: 2025-11-13
==============================================================================
"""

import random
import yaml
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
import numpy as np

class BankingDataGenerator:
    """
    This class creates synthetic banking transaction data.

    What it does:
    1. Creates realistic customer profiles
    2. Generates normal banking transactions
    3. Injects suspicious AML fraud patterns
    4. Adds text descriptions to transactions

    Think of it as a movie set designer - creating a realistic world for testing!
    """

    def __init__(self, spark, config):
        """
        Initialize the data generator.

        Args:
            spark: PySpark session (our connection to Spark)
            config: Configuration dictionary (our settings)
        """
        self.spark = spark
        self.config = config
        self.random_seed = 42
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        print("🏦 Banking Data Generator initialized!")
        print(f"📊 Will generate {config['data']['num_transactions']} transactions")
        print(f"⚠️  Fraud ratio: {config['data']['fraud_ratio']*100}%")

    def generate_transaction_data(self):
        """
        Main method to generate all transaction data.

        This is like the director calling "Action!" - it orchestrates everything!

        Returns:
            PySpark DataFrame with all transaction data
        """
        print("\n" + "="*80)
        print("🎬 STARTING DATA GENERATION")
        print("="*80)

        # Step 1: Generate customer profiles
        print("\n📋 Step 1: Creating customer profiles...")
        customers = self._generate_customers()

        # Step 2: Generate normal transactions
        print("\n💳 Step 2: Generating normal transactions...")
        normal_transactions = self._generate_normal_transactions(customers)

        # Step 3: Generate suspicious/fraud transactions
        print("\n🚨 Step 3: Injecting suspicious AML patterns...")
        fraud_transactions = self._generate_fraud_transactions(customers)

        # Step 4: Combine and shuffle
        print("\n🔀 Step 4: Combining and shuffling data...")
        all_transactions = normal_transactions.union(fraud_transactions)
        all_transactions = all_transactions.orderBy(rand(seed=self.random_seed))

        # Step 5: Add additional features
        print("\n✨ Step 5: Adding derived features...")
        final_data = self._add_derived_features(all_transactions)

        print("\n✅ Data generation complete!")
        print(f"📊 Total transactions: {final_data.count()}")
        print(f"⚠️  Suspicious transactions: {final_data.filter(col('is_suspicious') == 1).count()}")

        return final_data

    def _generate_customers(self):
        """
        Generate customer profiles.

        This creates our "cast of characters" - the customers in our banking system.
        Each customer has:
        - A unique ID
        - Risk level (low, medium, high)
        - Account creation date
        - Home country
        """
        num_customers = self.config['data']['num_customers']

        # Create customer data
        customers_data = []
        countries = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'SG', 'HK', 'JP', 'BR']
        risk_levels = ['low', 'medium', 'high']
        risk_weights = [0.7, 0.2, 0.1]  # Most customers are low risk

        for i in range(num_customers):
            customer = {
                'customer_id': f'CUST{i:06d}',
                'risk_level': random.choices(risk_levels, weights=risk_weights)[0],
                'country': random.choice(countries),
                'account_created_date': (datetime.now() - timedelta(days=random.randint(30, 1825))).strftime('%Y-%m-%d'),
                'customer_segment': random.choice(['retail', 'business', 'private_banking'])
            }
            customers_data.append(customer)

        # Create Spark DataFrame
        schema = StructType([
            StructField("customer_id", StringType(), False),
            StructField("risk_level", StringType(), False),
            StructField("country", StringType(), False),
            StructField("account_created_date", StringType(), False),
            StructField("customer_segment", StringType(), False)
        ])

        customers_df = self.spark.createDataFrame(customers_data, schema=schema)
        print(f"   ✓ Created {num_customers} customer profiles")

        return customers_df

    def _generate_normal_transactions(self, customers):
        """
        Generate normal (non-suspicious) banking transactions.

        These are everyday transactions: paying bills, shopping, transfers, etc.
        Think of this as the "normal life" of banking customers.
        """
        num_transactions = int(self.config['data']['num_transactions'] *
                              (1 - self.config['data']['fraud_ratio']))

        # Transaction types and their typical amounts
        transaction_types = {
            'ATM_WITHDRAWAL': (20, 500),
            'POS_PURCHASE': (5, 1000),
            'ONLINE_PURCHASE': (10, 500),
            'BILL_PAYMENT': (50, 1000),
            'TRANSFER': (100, 5000),
            'SALARY_DEPOSIT': (2000, 10000),
            'CHECK_DEPOSIT': (100, 5000)
        }

        # Text descriptions for transactions (realistic banking descriptions)
        transaction_descriptions = {
            'ATM_WITHDRAWAL': [
                'ATM withdrawal at branch',
                'Cash withdrawal ATM',
                'ATM cash out',
                'Automated teller withdrawal'
            ],
            'POS_PURCHASE': [
                'Purchase at grocery store',
                'Payment at restaurant',
                'Retail store purchase',
                'Gas station payment',
                'Shopping mall purchase'
            ],
            'ONLINE_PURCHASE': [
                'Online shopping transaction',
                'E-commerce payment',
                'Internet purchase',
                'Web store transaction'
            ],
            'BILL_PAYMENT': [
                'Utility bill payment',
                'Credit card payment',
                'Insurance premium',
                'Phone bill payment',
                'Internet service payment'
            ],
            'TRANSFER': [
                'Funds transfer to account',
                'Internal account transfer',
                'Transfer to savings',
                'P2P money transfer'
            ],
            'SALARY_DEPOSIT': [
                'Monthly salary deposit',
                'Payroll credit',
                'Salary payment received',
                'Employment income'
            ],
            'CHECK_DEPOSIT': [
                'Check deposit via mobile',
                'Bank check clearance',
                'Check payment received'
            ]
        }

        transactions_data = []
        customer_list = [row['customer_id'] for row in customers.select('customer_id').collect()]

        for i in range(num_transactions):
            # Pick a random customer
            customer_id = random.choice(customer_list)

            # Pick a transaction type
            trans_type = random.choice(list(transaction_types.keys()))
            amount_range = transaction_types[trans_type]

            # Generate amount (log-normal distribution for realistic amounts)
            amount = round(np.random.lognormal(
                mean=np.log(np.mean(amount_range)),
                sigma=0.5
            ), 2)
            amount = max(amount_range[0], min(amount, amount_range[1]))

            # Generate timestamp
            days_ago = random.randint(0, self.config['data']['date_range_days'])
            hours = random.randint(6, 23)  # Normal business hours mostly
            minutes = random.randint(0, 59)
            timestamp = datetime.now() - timedelta(days=days_ago, hours=hours, minutes=minutes)

            # Generate description
            description = random.choice(transaction_descriptions[trans_type])

            transaction = {
                'transaction_id': f'TXN{i:010d}',
                'customer_id': customer_id,
                'account_id': f'ACC{random.randint(1, self.config["data"]["num_accounts"]):06d}',
                'transaction_type': trans_type,
                'amount': amount,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'description': description,
                'channel': random.choice(['ATM', 'ONLINE', 'BRANCH', 'MOBILE']),
                'merchant_category': random.choice(['RETAIL', 'FOOD', 'UTILITIES', 'TRANSFER', 'CASH']),
                'is_suspicious': 0  # Normal transaction
            }
            transactions_data.append(transaction)

        # Create DataFrame
        schema = StructType([
            StructField("transaction_id", StringType(), False),
            StructField("customer_id", StringType(), False),
            StructField("account_id", StringType(), False),
            StructField("transaction_type", StringType(), False),
            StructField("amount", DoubleType(), False),
            StructField("timestamp", StringType(), False),
            StructField("description", StringType(), False),
            StructField("channel", StringType(), False),
            StructField("merchant_category", StringType(), False),
            StructField("is_suspicious", IntegerType(), False)
        ])

        normal_df = self.spark.createDataFrame(transactions_data, schema=schema)
        print(f"   ✓ Generated {num_transactions} normal transactions")

        return normal_df

    def _generate_fraud_transactions(self, customers):
        """
        Generate suspicious transactions with AML fraud patterns.

        These transactions exhibit red flags that money launderers use:
        1. Structuring: Breaking large amounts into smaller transactions
        2. Rapid movement: Moving money quickly through accounts
        3. Round numbers: Using suspiciously round amounts
        4. Unusual timing: Transactions at odd hours
        5. High-risk countries: Transactions to/from high-risk jurisdictions
        6. Smurfing: Multiple small deposits just below reporting thresholds

        Think of this as creating the "villains" in our story!
        """
        num_fraud = int(self.config['data']['num_transactions'] *
                       self.config['data']['fraud_ratio'])

        fraud_patterns = {
            'STRUCTURING': self._create_structuring_pattern,
            'RAPID_MOVEMENT': self._create_rapid_movement_pattern,
            'ROUND_AMOUNTS': self._create_round_amount_pattern,
            'UNUSUAL_TIMING': self._create_unusual_timing_pattern,
            'HIGH_RISK_COUNTRY': self._create_high_risk_country_pattern,
            'SMURFING': self._create_smurfing_pattern
        }

        fraud_transactions = []
        customer_list = [row['customer_id'] for row in customers.select('customer_id').collect()]

        # Distribute fraud across patterns
        pattern_names = list(fraud_patterns.keys())
        transactions_per_pattern = num_fraud // len(pattern_names)

        for pattern_name in pattern_names:
            print(f"   🔍 Creating {pattern_name} pattern...")
            pattern_func = fraud_patterns[pattern_name]

            for _ in range(transactions_per_pattern):
                customer_id = random.choice(customer_list)
                fraud_txn = pattern_func(customer_id, len(fraud_transactions))
                fraud_transactions.append(fraud_txn)

        # Create DataFrame
        schema = StructType([
            StructField("transaction_id", StringType(), False),
            StructField("customer_id", StringType(), False),
            StructField("account_id", StringType(), False),
            StructField("transaction_type", StringType(), False),
            StructField("amount", DoubleType(), False),
            StructField("timestamp", StringType(), False),
            StructField("description", StringType(), False),
            StructField("channel", StringType(), False),
            StructField("merchant_category", StringType(), False),
            StructField("is_suspicious", IntegerType(), False)
        ])

        fraud_df = self.spark.createDataFrame(fraud_transactions, schema=schema)
        print(f"   ✓ Generated {num_fraud} suspicious transactions")

        return fraud_df

    def _create_structuring_pattern(self, customer_id, base_idx):
        """
        Structuring: Breaking $50K into multiple $9,999 transactions
        This avoids the $10K reporting threshold!
        """
        amount = 9999.00  # Just under reporting threshold
        timestamp = datetime.now() - timedelta(days=random.randint(0, 365))

        return {
            'transaction_id': f'TXN{base_idx:010d}',
            'customer_id': customer_id,
            'account_id': f'ACC{random.randint(1, self.config["data"]["num_accounts"]):06d}',
            'transaction_type': 'TRANSFER',
            'amount': amount,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'Large cash deposit structured payment',
            'channel': random.choice(['BRANCH', 'WIRE']),
            'merchant_category': 'TRANSFER',
            'is_suspicious': 1
        }

    def _create_rapid_movement_pattern(self, customer_id, base_idx):
        """
        Rapid Movement: Money moves through account very quickly
        In-and-out within hours!
        """
        amount = round(random.uniform(5000, 20000), 2)
        timestamp = datetime.now() - timedelta(days=random.randint(0, 365),
                                                hours=random.randint(0, 2))

        return {
            'transaction_id': f'TXN{base_idx:010d}',
            'customer_id': customer_id,
            'account_id': f'ACC{random.randint(1, self.config["data"]["num_accounts"]):06d}',
            'transaction_type': 'TRANSFER',
            'amount': amount,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'Rapid funds movement layering transaction',
            'channel': 'ONLINE',
            'merchant_category': 'TRANSFER',
            'is_suspicious': 1
        }

    def _create_round_amount_pattern(self, customer_id, base_idx):
        """
        Round Amounts: Suspiciously round numbers
        Real people rarely transfer exactly $50,000.00
        """
        round_amounts = [10000, 25000, 50000, 100000, 500000]
        amount = float(random.choice(round_amounts))
        timestamp = datetime.now() - timedelta(days=random.randint(0, 365))

        return {
            'transaction_id': f'TXN{base_idx:010d}',
            'customer_id': customer_id,
            'account_id': f'ACC{random.randint(1, self.config["data"]["num_accounts"]):06d}',
            'transaction_type': 'WIRE_TRANSFER',
            'amount': amount,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'Large round amount wire transfer suspicious',
            'channel': 'WIRE',
            'merchant_category': 'TRANSFER',
            'is_suspicious': 1
        }

    def _create_unusual_timing_pattern(self, customer_id, base_idx):
        """
        Unusual Timing: Transactions at 3 AM
        Who transfers money at weird hours?
        """
        amount = round(random.uniform(1000, 15000), 2)
        timestamp = datetime.now() - timedelta(days=random.randint(0, 365),
                                                hours=random.randint(0, 5))  # Midnight to 5 AM

        return {
            'transaction_id': f'TXN{base_idx:010d}',
            'customer_id': customer_id,
            'account_id': f'ACC{random.randint(1, self.config["data"]["num_accounts"]):06d}',
            'transaction_type': 'ONLINE_TRANSFER',
            'amount': amount,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'Unusual hours online transfer suspicious activity',
            'channel': 'ONLINE',
            'merchant_category': 'TRANSFER',
            'is_suspicious': 1
        }

    def _create_high_risk_country_pattern(self, customer_id, base_idx):
        """
        High-Risk Country: Transfers to countries with weak AML controls
        """
        amount = round(random.uniform(5000, 50000), 2)
        timestamp = datetime.now() - timedelta(days=random.randint(0, 365))

        return {
            'transaction_id': f'TXN{base_idx:010d}',
            'customer_id': customer_id,
            'account_id': f'ACC{random.randint(1, self.config["data"]["num_accounts"]):06d}',
            'transaction_type': 'INTERNATIONAL_WIRE',
            'amount': amount,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'International wire transfer high risk jurisdiction offshore',
            'channel': 'WIRE',
            'merchant_category': 'INTERNATIONAL',
            'is_suspicious': 1
        }

    def _create_smurfing_pattern(self, customer_id, base_idx):
        """
        Smurfing: Multiple small deposits to avoid detection
        Think of many "smurfs" each carrying small amounts
        """
        amount = round(random.uniform(500, 2000), 2)
        timestamp = datetime.now() - timedelta(days=random.randint(0, 365))

        return {
            'transaction_id': f'TXN{base_idx:010d}',
            'customer_id': customer_id,
            'account_id': f'ACC{random.randint(1, self.config["data"]["num_accounts"]):06d}',
            'transaction_type': 'CASH_DEPOSIT',
            'amount': amount,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'Multiple small cash deposits smurfing pattern',
            'channel': 'BRANCH',
            'merchant_category': 'CASH',
            'is_suspicious': 1
        }

    def _add_derived_features(self, transactions_df):
        """
        Add derived features that will help with clustering.

        These are like "calculated fields" - we compute them from existing data:
        - Hour of day
        - Day of week
        - Amount in log scale (for better distribution)
        - Is it a round number?
        - Account balance (simulated)
        """
        enhanced_df = transactions_df \
            .withColumn('timestamp_dt', to_timestamp(col('timestamp'))) \
            .withColumn('transaction_hour', hour(col('timestamp_dt'))) \
            .withColumn('transaction_day_of_week', dayofweek(col('timestamp_dt'))) \
            .withColumn('transaction_month', month(col('timestamp_dt'))) \
            .withColumn('amount_log', log10(col('amount') + 1)) \
            .withColumn('is_round_amount',
                       when((col('amount') % 1000 == 0) & (col('amount') >= 1000), 1).otherwise(0)) \
            .withColumn('account_balance',
                       round(col('amount') * (1 + rand(seed=42) * 10), 2)) \
            .withColumn('is_high_value',
                       when(col('amount') >= 10000, 1).otherwise(0))

        return enhanced_df

    def save_data(self, df, path):
        """
        Save the generated data to disk.

        We use Parquet format - it's like a compressed, efficient filing cabinet
        for big data!
        """
        print(f"\n💾 Saving data to: {path}")
        df.write.mode('overwrite').parquet(path)
        print("   ✓ Data saved successfully!")


def main():
    """
    Main function to run the data generator standalone.
    This is like the "main entrance" to our program!
    """
    print("\n" + "="*80)
    print("🏦 BANKING TRANSACTION DATA GENERATOR")
    print("="*80)

    # Initialize Spark
    print("\n⚙️  Initializing Spark...")
    spark = SparkSession.builder \
        .appName("AML_Data_Generator") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print("   ✓ Spark initialized successfully!")

    # Load configuration
    print("\n📋 Loading configuration...")
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("   ✓ Configuration loaded!")

    # Generate data
    generator = BankingDataGenerator(spark, config)
    data = generator.generate_transaction_data()

    # Show sample data
    print("\n📊 Sample of generated data:")
    data.show(10, truncate=False)

    print("\n📈 Data schema:")
    data.printSchema()

    # Show some statistics
    print("\n📊 Data Statistics:")
    print(f"   Total transactions: {data.count()}")
    print(f"   Suspicious transactions: {data.filter(col('is_suspicious') == 1).count()}")
    print(f"   Normal transactions: {data.filter(col('is_suspicious') == 0).count()}")

    # Save data
    output_path = f"../{config['data']['raw_data_path']}"
    generator.save_data(data, output_path)

    print("\n" + "="*80)
    print("✅ DATA GENERATION COMPLETE!")
    print("="*80)

    spark.stop()


if __name__ == "__main__":
    main()

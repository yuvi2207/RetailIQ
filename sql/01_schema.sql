-- ============================================================================
-- RetailIQ - PostgreSQL Database Schema
-- ============================================================================

-- ============================================================================
-- CUSTOMERS
-- ============================================================================

CREATE TABLE customers (
    customer_id VARCHAR(32) PRIMARY KEY,
    customer_unique_id VARCHAR(32) NOT NULL,
    customer_zip_code_prefix VARCHAR(5),
    customer_city TEXT,
    customer_state CHAR(2)
);

CREATE INDEX idx_customers_unique_id
    ON customers(customer_unique_id);

CREATE INDEX idx_customers_state
    ON customers(customer_state);


-- ============================================================================
-- PRODUCTS
-- ============================================================================

CREATE TABLE products (
    product_id VARCHAR(32) PRIMARY KEY,
    product_category_name TEXT,
    product_name_length INTEGER,
    product_description_length INTEGER,
    product_photos_qty INTEGER,
    product_weight_g NUMERIC(10,2),
    product_length_cm NUMERIC(10,2),
    product_height_cm NUMERIC(10,2),
    product_width_cm NUMERIC(10,2)
);


-- ============================================================================
-- SELLERS
-- ============================================================================

CREATE TABLE sellers (
    seller_id VARCHAR(32) PRIMARY KEY,
    seller_zip_code_prefix VARCHAR(5),
    seller_city TEXT,
    seller_state CHAR(2)
);

CREATE INDEX idx_sellers_state
    ON sellers(seller_state);


-- ============================================================================
-- CATEGORIES
-- ============================================================================

CREATE TABLE categories (
    category_name_portuguese TEXT PRIMARY KEY,
    category_name_english TEXT
);


-- ============================================================================
-- ORDERS
-- ============================================================================

CREATE TABLE orders (
    order_id VARCHAR(32) PRIMARY KEY,
    customer_id VARCHAR(32) NOT NULL,
    order_status TEXT NOT NULL,

    order_purchase_timestamp TIMESTAMP NOT NULL,
    order_approved_at TIMESTAMP,
    order_delivered_carrier_date TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP NOT NULL,

    CONSTRAINT fk_orders_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
);

CREATE INDEX idx_orders_customer_id
    ON orders(customer_id);

CREATE INDEX idx_orders_purchase_timestamp
    ON orders(order_purchase_timestamp);

CREATE INDEX idx_orders_status
    ON orders(order_status);


-- ============================================================================
-- ORDER ITEMS
-- ============================================================================

CREATE TABLE order_items (
    order_id VARCHAR(32) NOT NULL,
    order_item_id INTEGER NOT NULL,
    product_id VARCHAR(32) NOT NULL,
    seller_id VARCHAR(32) NOT NULL,

    shipping_limit_date TIMESTAMP,
    price NUMERIC(12,2) NOT NULL,
    freight_value NUMERIC(12,2) NOT NULL,

    PRIMARY KEY (order_id, order_item_id),

    CONSTRAINT fk_items_order
        FOREIGN KEY (order_id)
        REFERENCES orders(order_id),

    CONSTRAINT fk_items_product
        FOREIGN KEY (product_id)
        REFERENCES products(product_id),

    CONSTRAINT fk_items_seller
        FOREIGN KEY (seller_id)
        REFERENCES sellers(seller_id),

    CONSTRAINT chk_item_price
        CHECK (price >= 0),

    CONSTRAINT chk_freight_value
        CHECK (freight_value >= 0)
);

CREATE INDEX idx_order_items_product
    ON order_items(product_id);

CREATE INDEX idx_order_items_seller
    ON order_items(seller_id);


-- ============================================================================
-- PAYMENTS
-- ============================================================================

CREATE TABLE payments (
    order_id VARCHAR(32) NOT NULL,
    payment_sequential INTEGER NOT NULL,
    payment_type TEXT NOT NULL,
    payment_installments INTEGER NOT NULL,
    payment_value NUMERIC(12,2) NOT NULL,

    PRIMARY KEY (order_id, payment_sequential),

    CONSTRAINT fk_payments_order
        FOREIGN KEY (order_id)
        REFERENCES orders(order_id),

    CONSTRAINT chk_payment_value
        CHECK (payment_value >= 0),

    CONSTRAINT chk_payment_installments
        CHECK (payment_installments >= 0)
);

CREATE INDEX idx_payments_order
    ON payments(order_id);

CREATE INDEX idx_payments_type
    ON payments(payment_type);


-- ============================================================================
-- REVIEWS
-- ============================================================================

CREATE TABLE reviews (
    review_record_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    review_id VARCHAR(32) NOT NULL,
    order_id VARCHAR(32) NOT NULL,

    review_score SMALLINT,
    review_comment_title TEXT,
    review_comment_message TEXT,
    review_creation_date TIMESTAMP,
    review_answer_timestamp TIMESTAMP,

    CONSTRAINT fk_reviews_order
        FOREIGN KEY (order_id)
        REFERENCES orders(order_id),

    CONSTRAINT chk_review_score
        CHECK (review_score IS NULL OR review_score BETWEEN 1 AND 5)
);

CREATE INDEX idx_reviews_order
    ON reviews(order_id);

CREATE INDEX idx_reviews_review_id
    ON reviews(review_id);


-- ============================================================================
-- GEOLOCATION
-- ============================================================================

CREATE TABLE geolocation (
    geolocation_zip_code_prefix VARCHAR(5) NOT NULL,
    geolocation_lat NUMERIC(10,7),
    geolocation_lng NUMERIC(10,7),
    geolocation_city TEXT,
    geolocation_state CHAR(2)
);

CREATE INDEX idx_geolocation_zip
    ON geolocation(geolocation_zip_code_prefix);


-- ============================================================================
-- SCHEMA COMPLETE
-- ============================================================================
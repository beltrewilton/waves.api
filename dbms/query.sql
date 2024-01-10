--SELECT_PRODUCT
SELECT
    p.id,
    p.name,
    p.cost,
    pl.price,
    p.margin,
    p.code,
    p.img_path,
    p.date_create,
    p.active,
    p.image_raw,
    i.quantity,
    s.name as store_name
  FROM product  p, app_inventory i, app_store s, pricing_list pl, pricing pc
WHERE
      p.id = i.product_id
  AND s.id = i.store_id
  AND pl.pricing_id = pc.id
  AND pl.product_id  = i.product_id
  AND s.name = %s
  AND pc.price_key = 'DEFAULT'
  AND p.active = 1
  AND p.archived = 0
  ;

--SELECT_ALL_PRODUCT_ACTIVE
SELECT
    p.id,
    p.name,
    p.cost,
    p.price,
    p.margin,
    p.code,
    p.img_path,
    p.date_create,
    p.active,
    p.image_raw
  FROM product  p
    WHERE p.active = 1
     AND  p.archived = 0
  ORDER BY p.id DESC
;

--SELECT_ALL_PRODUCT
SELECT
    p.id,
    p.name,
    p.cost,
    p.price,
    p.margin,
    p.code,
    p.img_path,
    p.date_create,
    p.active,
    p.image_raw
  FROM product  p
   WHERE p.archived = 0
  ORDER BY p.id DESC
;

--SELECT_ALL_PRODUCT_DATE_ORDERED
SELECT
    p.id,
    p.name,
    p.cost,
    p.price,
    p.margin,
    p.code,
    p.img_path,
    p.date_create,
    p.active,
    p.image_raw,
    (SELECT  i.last_update
       FROM app_inventory i
       WHERE
             i.product_id = p.id
	     AND i.last_update = (SELECT MAX(x.last_update)
                                  FROM app_inventory x
								WHERE x.store_id = %s
								  AND  x.product_id = p.id
                                 LIMIT 1)
      LIMIT 1
	)
    AS last_update_inv
  FROM product  p
   WHERE
        p.active = 1
    AND p.archived = 0
  ORDER BY COALESCE(last_update_inv, p.date_create) DESC
;

--SELECT_ALL_PRODUCT_BY_ID
SELECT
    p.id,
    p.name,
    p.cost,
    p.price,
    p.margin,
    p.code,
    p.img_path,
    p.date_create,
    p.active,
    p.image_raw
  FROM product  p
  WHERE p.id = %s
;

--INSERT_PRODUCT
INSERT INTO product (name, cost, code, date_create, user_modified, image_raw, active)
 VALUES (%s, %s, %s, NOW(), %s, %s, 1)
;

--INSERT_PRODUCT_INVENTORY
INSERT INTO app_inventory (quantity, product_id, store_id)
   VALUES (%s, %s, %s)
;

--INSERT_PRUDUCT_PRICING
INSERT INTO pricing_list (price, user_modified, product_id, pricing_id)
   VALUES (%s, %s, %s, %s)
;

--SELECT_PRICING_LIST
SELECT
     pl.id,
     pl.price,
     pl.user_modified,
     pl.date_create,
     p.price_key,
     p.label
FROM pricing_list pl, pricing p
  WHERE pl.pricing_id = p.id
   AND p.status = 1
   AND  pl.product_id = %s
ORDER BY pl.id
;

--SELECT_PRICING_LABELS
SELECT
    p.id,
    p.price_key,
    p.label
FROM pricing p
WHERE
   p.status = 1
;

--SELECT_PRICING
SELECT
    p.id,
	p.label,
    p.price_key,
    p.date_create,
    p.status,
    p.user_modified
FROM pricing p
;

--INSERT_PRICING
INSERT INTO pricing (label, price_key, user_modified)
  VALUES (%s, %s, %s)
;

--INSERT_PRICING_LIST
INSERT INTO pricing_list (price, user_modified, product_id, pricing_id)
    SELECT (p.price * %s), %s, p.id, %s FROM product p
;

--SELECT_SALES_BY_CLIENT
SELECT
	s.id,
    s.amount,
    s.sub,
    s.discount,
    s.tax_amount,
    s.delivery_charge,
    s.sequence,
    s.sequence_type,
    s.status,
    s.sale_type,
    s.date_create,
    s.login,
    s.additional_info,
    cli.id as client_id,
    cli.name as client_name,
    cli.document_id,
    cli.celphone,
    cli.address,
    (SELECT
        SUM(paid.amount)
	   FROM sale_paid paid
      WHERE  paid.sale_id = s.id
      GROUP BY paid.sale_id
	) as total_paid,
    (SELECT
         COUNT(*)
	  FROM sale_line line
     WHERE line.sale_id = s.id
     GROUP BY line.sale_id
    ) as total_line,

	(CASE
      WHEN s.status = 'RETURN' THEN 'cancelled'
      WHEN (s.amount - IFNULL((SELECT
							SUM(paid.amount)
						   FROM sale_paid paid
						  WHERE  paid.sale_id = s.id
						  GROUP BY paid.sale_id
						), 0)
		   ) > 0 THEN 'open'
      ELSE 'close'
	 END) as invoice_status
 FROM sale s,  app_store store, client cli
WHERE s.client_id = cli.id
  AND s.store_id = store.id
  AND store.name = %s
  AND s.date_create BETWEEN %s AND %s
  AND (CASE
      WHEN s.status = 'RETURN' THEN 'cancelled'
      WHEN (s.amount - IFNULL((SELECT
							SUM(paid.amount)
						   FROM sale_paid paid
						  WHERE  paid.sale_id = s.id
						  GROUP BY paid.sale_id
						), 0)
		   ) > 0 THEN 'open'
      ELSE 'close'
	 END) in %s
  AND cli.id = %s
  AND s.login REGEXP %s
ORDER BY s.id DESC
;

--SELECT_SALES_BY_INVOICE_STATUS
SELECT
	s.id,
    s.amount,
    s.sub,
    s.discount,
    s.tax_amount,
    s.delivery_charge,
    s.sequence,
    s.sequence_type,
    s.status,
    s.sale_type,
    s.date_create,
    s.login,
    s.additional_info,
    cli.id as client_id,
    cli.name as client_name,
    cli.document_id,
    cli.celphone,
    cli.address,
    (SELECT
        SUM(paid.amount)
	   FROM sale_paid paid
      WHERE  paid.sale_id = s.id
      GROUP BY paid.sale_id
	) as total_paid,
    (SELECT
         COUNT(*)
	  FROM sale_line line
     WHERE line.sale_id = s.id
     GROUP BY line.sale_id
    ) as total_line,

	(CASE
      WHEN s.status = 'RETURN' THEN 'cancelled'
      WHEN (s.amount - IFNULL((SELECT
							SUM(paid.amount)
						   FROM sale_paid paid
						  WHERE  paid.sale_id = s.id
						  GROUP BY paid.sale_id
						), 0)
		   ) > 0 THEN 'open'
      ELSE 'close'
	 END) as invoice_status
 FROM sale s,  app_store store, client cli
WHERE s.client_id = cli.id
  AND s.store_id = store.id
  AND store.name = %s
   AND s.date_create BETWEEN %s AND %s
  AND (CASE
      WHEN s.status = 'RETURN' THEN 'cancelled'
      WHEN (s.amount - IFNULL((SELECT
							SUM(paid.amount)
						   FROM sale_paid paid
						  WHERE  paid.sale_id = s.id
						  GROUP BY paid.sale_id
						), 0)
		   ) > 0 THEN 'open'
      ELSE 'close'
	 END) in %s
  AND s.login REGEXP %s
ORDER BY s.id DESC
;

--SELECT_LINE
SELECT
	sl.amount as line_amount,
    sl.tax_amount as line_tax_amount,
    sl.discount as line_discount,
    sl.quantity,
    sl.total_amount,
	p.id as product_id,
    p.name as product_name,
    p.cost as product_cost,
    p.price as product_price,
    p.active
FROM sale_line sl, product p
WHERE sl.product_id = p.id
  AND sl.sale_id = %s;

--SELECT_PAID
SELECT
	sp.id as paid_id,
    sp.amount as paid_amount,
    sp.type as paid_type,
    sp.date_create as paid_date_create
FROM sale_paid sp
WHERE sp.sale_id = %s
;

--INSERT_PAID
INSERT INTO sale_paid(amount, type, sale_id, date_create)
   VALUES(%s, %s, %s, NOW())
;

--UPDATE_SALES_AS_RETURN
UPDATE sale
  SET status = 'RETURN'
WHERE id = %s
;

--SELECT_STORE_FROM_SALE
SELECT s.store_id
 FROM sale s
WHERE
  s.id = %s
;

--SELECT_SALE_LINES_RETURN
SELECT sl.quantity, sl.product_id
 FROM sale_line sl
WHERE
  sl.sale_id = %s
;

--SALE_RETURN_QUANTITY
UPDATE app_inventory
 SET quantity = quantity + %s
 WHERE
       product_id = %s
   AND store_id = %s
;

--SELECT_STORES
 SELECT
      s.id,
      s.name,
      s.company_id,
      s.slogan,
      s.logo,
      s.address
FROM app_store s
 ORDER BY s.id
;

--SELECT_INV_HEAD_BYSTOREID
SELECT
     i.id,
     i.name,
     i.date_create,
     i.date_close,
     i.status,
     i.memo
FROM
    app_inventory_head i
WHERE
    i.store_id = %s
AND i.status = 0
;


--SELECT_PRODUCT_INV
 SELECT
      i.id,
      i.prev_quantity,
      i.quantity,
      i.next_quantity,
      i.status,
      st.name,
      st.id as store_id
 FROM app_inventory i, app_store st
  WHERE i.store_id = st.id
   AND  i.product_id = %s
   AND  st.name = %s
;

--SELECT_PRODUCT_ALL_INV
 SELECT
      i.id,
      i.prev_quantity,
      i.quantity,
      i.next_quantity,
      i.status,
      st.name,
      st.id as store_id
 FROM app_inventory i, app_store st
  WHERE i.store_id = st.id
   AND  i.product_id = %s
;

--SELECT_INVENTORY_HEAD
SELECT
	h.id,
    h.name,
    h.date_create,
    h.date_close,
    h.status,
    h.memo,
    st.id as store_id,
    st.name as store_name
 FROM app_inventory_head h, app_store st
	WHERE h.store_id = st.id
	AND st.name = %s
	AND h.status = 0
;

--INSERT_INVENTORY_HEAD
INSERT INTO app_inventory_head (name, memo, store_id)
 VALUES (%s, %s, %s)
;

--PREPARE_FOR_INVENTORY
UPDATE
     app_inventory
  SET
     next_quantity = quantity,
     prev_quantity = quantity,
     status = 'waiting'
WHERE
     store_id = %s
;

--REORDER_INVENTORY
UPDATE
     app_inventory
  SET
     quantity = next_quantity,
     status = 'quiet'
WHERE
     store_id = %s
 AND status = 'changed'
;

--UPDATE_INVENTORY_NEXT
UPDATE
     app_inventory
  SET
     next_quantity = %s,
     user_updated = %s,
     last_update = NOW(),
     status = 'changed'
  WHERE
        product_id = %s
   AND  store_id = %s
;

--CLOSE_INVENTORY_HEAD
UPDATE
     app_inventory_head
SET
	 date_close = NOW(),
     status = 1
WHERE
	store_id = %s
ORDER BY date_create DESC
 LIMIT 1
;

--CLOSE_INVENTORY_HEAD_CANCEL
UPDATE
     app_inventory_head
SET
	 date_close = NOW(),
     status = 1,
     memo = CONCAT('CANCELLED: ', memo)
WHERE
	store_id = %s
ORDER BY date_create DESC
 LIMIT 1
;

--CLOSE_INVENTORY_CANCEL
UPDATE
     app_inventory
  SET
     status = 'quiet'
WHERE
     store_id = %s
 AND status = 'changed'
;

--SELECT_INV_VALUATION
SELECT
     SUM(p.cost * i.quantity) AS inv_valuation
FROM app_inventory i, app_store st, product p
 WHERE
	  st.id = i.store_id
  AND i.product_id = p.id
  AND p.active = 1
  AND p.archived = 0
  AND i.quantity > 0
  AND st.name = %s
;

--SELECT_INV_VALUATION_CHANGED
SELECT
     SUM(p.cost * i.next_quantity) AS inv_valuation_changed,
     COUNT(*) AS count_inv_valuation_changed
FROM app_inventory i, app_store st, product p
 WHERE
	  st.id = i.store_id
  AND i.product_id = p.id
  AND p.active = 1
  AND p.archived = 0
  AND i.quantity > 0
  AND i.status = 'changed'
  AND st.name = %s
;

--INSERT_APP_STORE
INSERT INTO app_store (name) VALUES (%s)
;

--INSERT_APP_INV_STORE
INSERT INTO app_inventory (quantity, product_id, store_id)
 SELECT 0, id, %s FROM product
;

--SELECT_FROM_PRODUCT_ORDER
SELECT
       (SELECT
			  b.id
		FROM
			 bulk_order b, bulk_order_line bl
		WHERE
			 bl.bulk_order_id  = b.id
		 AND bl.product_order_id = o.id
       ) AS bulk_order_id,
       (SELECT
			  b.name
		FROM
			 bulk_order b, bulk_order_line bl
		WHERE
			 bl.bulk_order_id  = b.id
		 AND bl.product_order_id = o.id
       ) AS bulk_order_name,
       (SELECT
			  b.memo
		FROM
			 bulk_order b, bulk_order_line bl
		WHERE
			 bl.bulk_order_id  = b.id
		 AND bl.product_order_id = o.id
       ) AS bulk_order_memo,
       o.id,
       o.name,
       o.memo,
       o.order_type,
       o.status,
       o.user_requester,
       o.user_receiver,
       o.date_opened,
       o.date_closed,
       o.from_origin_id,
       o.to_store_id,
       (CASE
          WHEN o.order_type = 'purchase' THEN
               (SELECT v.name FROM provider v WHERE v.id = o.from_origin_id)
          ELSE
               (SELECT s.name FROM app_store s WHERE s.id = o.from_origin_id)
       END) AS from_store_name,
       (
         SELECT s.name FROM app_store s WHERE s.id = o.to_store_id
       ) AS to_store_name,
       (
         SELECT COUNT(*)
         FROM product_order_line l
         WHERE l.product_order_id = o.id
       ) AS products_in_order,
       (
         SELECT COUNT(*)
         FROM product_order_line l
         WHERE l.product_order_id = o.id
          AND l.status = 'issue'
       ) AS products_in_order_issue,
       (
         SELECT
            SUM(l.quantity_observed * p.cost)
        FROM product_order_line l, product p
        WHERE
            l.product_id = p.id
        AND l.product_order_id = o.id
       ) AS value_in_order
 FROM  product_order o
   WHERE order_type = %s
 ORDER BY o.id DESC
;

--SELECT_FROM_PRODUCT_ORDER_BYID
SELECT
        (SELECT
			  b.id
		FROM
			 bulk_order b, bulk_order_line bl
		WHERE
			 bl.bulk_order_id  = b.id
		 AND bl.product_order_id = o.id
       ) AS bulk_order_id,
       (SELECT
			  b.name
		FROM
			 bulk_order b, bulk_order_line bl
		WHERE
			 bl.bulk_order_id  = b.id
		 AND bl.product_order_id = o.id
       ) AS bulk_order_name,
       (SELECT
			  b.memo
		FROM
			 bulk_order b, bulk_order_line bl
		WHERE
			 bl.bulk_order_id  = b.id
		 AND bl.product_order_id = o.id
       ) AS bulk_order_memo,
       o.id,
       o.name,
       o.memo,
       o.order_type,
       o.status,
       o.user_requester,
       o.user_receiver,
       o.date_opened,
       o.date_closed,
       o.from_origin_id,
       o.to_store_id,
       (CASE
          WHEN o.order_type = 'purchase' THEN
               (SELECT v.name FROM provider v WHERE v.id = o.from_origin_id)
          ELSE
               (SELECT s.name FROM app_store s WHERE s.id = o.from_origin_id)
       END) AS from_store_name,
       (
         SELECT s.name FROM app_store s WHERE s.id = o.to_store_id
       ) AS to_store_name,
       (
         SELECT COUNT(*)
         FROM product_order_line l
         WHERE l.product_order_id = o.id
       ) AS products_in_order,
       (
         SELECT COUNT(*)
         FROM product_order_line l
         WHERE l.product_order_id = o.id
          AND l.status = 'issue'
       ) AS products_in_order_issue,
       (
         SELECT
            SUM(l.quantity_observed * p.cost)
        FROM product_order_line l, product p
        WHERE
            l.product_id = p.id
        AND l.product_order_id = o.id
       ) AS value_in_order
 FROM  product_order o
  WHERE o.id = %s
   AND order_type = %s
;

--SELECT_FROM_PRODUCT_ORDER_LINE
SELECT
       h.id,
       h.product_id,
       h.from_origin_id,
       h.to_store_id,
       h.product_order_id,
       h.quantity,
       h.quantity_observed,
       h.status,
       h.date_create,
       p.name AS product_name,
       p.cost,
       p.code,
       p.active
 FROM  product_order_line h, product p
 WHERE
        h.product_order_id = %s
   AND  h.product_id = p.id
 ORDER BY h.id DESC
;

--SELECT_FROM_PRODUCT_ORDER_LINE_BYID
SELECT
       h.id,
       h.product_id,
       h.from_origin_id,
       h.to_store_id,
       h.product_order_id,
       h.quantity,
       h.quantity_observed,
       h.status,
       h.date_create
 FROM  product_order_line h
 WHERE
     h.id = %s
;

--SELECT_FROM_PRODUCT_ORDER_LINE_BYARGS
SELECT
       h.id,
       h.quantity,
       h.status
 FROM  product_order_line h
 WHERE
     h.product_order_id = %s
 AND h.product_id = %s
;

--INSERT_PRODUCT_ORDER
INSERT INTO product_order (name, memo, order_type, user_requester, from_origin_id, to_store_id)
  VALUES (%s, %s, %s, %s, %s, %s)
;

--INSERT_PRODUCT_ORDER_LINE
INSERT INTO product_order_line (product_id, from_origin_id, to_store_id, product_order_id, quantity, quantity_observed)
   VALUES (%s, %s, %s, %s, %s, %s)
;

--VALIDATE_PRODUCT_ORDER_LINE_EXIST
SELECT
		COUNT(*) AS line_count
FROM product_order_line l
WHERE
    l.product_order_id = %s
AND l.product_id = %s
;

--UPDATE_PRODUCT_ORDER_LINE
UPDATE product_order_line
  SET
      quantity = %s,
      quantity_observed = %s,
      status = %s
WHERE
	product_order_id = %s
AND product_id = %s
;

--UPDATE_APPROBAL_ISSUE_ORDER_LINE
UPDATE product_order_line
  SET
      quantity = quantity_observed,
      status = 'pending'
WHERE
	product_order_id = %s
AND product_id = %s
;

--DELETE_PRODUCT_ORDER_LINE
DELETE FROM product_order_line
WHERE
	product_order_id = %s
AND product_id = %s
;

--SUBSTRACT_FROM_STORE
UPDATE app_inventory
 SET prev_quantity = quantity,
     quantity = quantity - %s,
     last_update = NOW(),
     user_updated = %s
 WHERE store_id = %s
  AND  product_id = %s
;

--SUBSTRACT_FROM_STORE_DONTTOUCH_PREV_QUANTITY
UPDATE app_inventory
 SET
     quantity = quantity - %s,
     last_update = NOW(),
     user_updated = %s
 WHERE store_id = %s
  AND  product_id = %s
;

--PROCESS_APP_INVENTORY
UPDATE app_inventory i,
  (
    SELECT
            l.id,
            l.to_store_id,
            l.product_id,
            (CASE
              WHEN l.status = 'issue' THEN l.quantity_observed
              ELSE l.quantity
             END) as abs_quantity
	FROM product_order_line l
    WHERE
		 l.product_order_id = %s
  ) AS line
  SET
	   i.prev_quantity = i.quantity,
       i.user_updated = %s,
       i.last_update = NOW(),
	   i.quantity = i.quantity + line.abs_quantity
 WHERE
       i.store_id = line.to_store_id
   AND i.product_id = line.product_id
;

--PROCESS_APP_INVENTORY_ISSUE_BACK
UPDATE app_inventory i,
  (
    SELECT
            l.id,
            l.from_origin_id,
            l.product_id,
            l.quantity_observed,
            l.quantity
	FROM product_order_line l
    WHERE
		 l.product_order_id = %s
	AND  l.status = 'issue'
  ) AS line
  SET
       i.user_updated = %s,
       i.last_update = NOW(),
	   i.quantity = i.quantity + (line.quantity - line.quantity_observed)
 WHERE
       i.store_id = line.from_origin_id
   AND i.product_id = line.product_id
;

--ROLLBACK_APP_INVENTORY
UPDATE app_inventory i,
  (
    SELECT
            l.id,
            l.from_origin_id,
            l.product_id,
            l.quantity
	FROM product_order_line l
    WHERE
		 l.product_order_id = %s
  ) AS line
  SET
       i.user_updated = %s,
       i.last_update = NOW(),
	   i.quantity = i.quantity + line.quantity
 WHERE
       i.store_id = line.from_origin_id
   AND i.product_id = line.product_id
;

--CANCEL_ORDER_LINE
UPDATE product_order_line l
  SET
       l.status = 'cancelled'
 WHERE
       l.product_order_id = %s
;

--CANCEL_ORDER
UPDATE product_order o
  SET
       o.memo = CONCAT(o.memo, %s),
       o.date_closed = NOW(),
       o.status = 'cancelled'
 WHERE
       o.id = %s
;

--UPDATE_ORDER_LINE_PROCESS
UPDATE product_order_line l
  SET l.status = 'transfered'
WHERE
	l.product_order_id = %s
;

--UPDATE_PRODUCT_ORDER_PROCESS
UPDATE product_order o
   SET o.user_receiver = %s,
       o.date_closed = NOW(),
       o.status = 'closed'
WHERE
      o.id = %s
;

--UPDATE_PRODUCT_ORDER_LINE_ISSUE_COUNT
UPDATE product_order_line
  SET
      quantity_observed = %s,
      user_receiver = %s,
      receiver_memo = %s,
      status = %s,
      receiver_last_update = NOW()
WHERE
	product_order_id = %s
AND product_id = %s
;

--APP_INVENTORY_REMAIN_QUANTITY
SELECT
      i.quantity
FROM app_inventory i
WHERE
      i.store_id = %s
  AND i.product_id = %s
;

--SELECT_PROVIDERS
SELECT g.id,
       g.name
  FROM provider g
ORDER BY g.id
;

--CHECK_FOR_ORDER_IN_BULK
SELECT
       bl.id
  FROM bulk_order_line bl
WHERE bl.product_order_id = %s
;

--UPDATE_BULK_ORDER_LINE
UPDATE
      bulk_order_line bl
  SET
      bl.bulk_order_id = %s
WHERE
      bl.product_order_id = %s
;

--INSERT_BULK_ORDER_LINE
INSERT INTO bulk_order_line (bulk_order_id, product_order_id)
  VALUES (%s, %s)
;

--INSERT_BULK_ORDER
INSERT INTO bulk_order (name, memo, user_create)
  VALUES (%s, %s, %s)
;


--SELECT_BULK_ORDER
SELECT
      bl.id,
      bl.name,
      bl.memo,
      bl.date_create,
      bl.user_create
  FROM bulk_order bl
ORDER BY bl.id DESC
;

--SELECT_USER
SELECT
      u.id,
      u.username,
      u.password,
      u.first_name,
      u.last_name,
      u.is_active,
      u.date_joined,
      u.last_login,
      u.pic
FROM
     users u
WHERE
    u.username = %s
 ;

--SELECT_USER_STORE
SELECT
      s.id,
      s.name,
      s.company_id,
      s.slogan,
      s.logo,
      s.address
FROM
     app_user_store x, app_store s
WHERE
     x.user_id = %s
AND  s.id = x.store_id
;

--SELECT_USER_SCOPES
SELECT
      s.name as scope
FROM
     scopes s
WHERE
     s.user_id = %s
;

--UPDATE_SEQUENCE
UPDATE app_sequence
   SET current_seq = current_seq + increment_by
  WHERE code = %s;

--SELECT_SEQ
SELECT
    s.prefix,
    s.fill,
    s.increment_by,
    s.current_seq
 FROM app_sequence s
   WHERE s.code = %s
;

--SELECT_SEQ_ALL
SELECT
    s.id,
    s.name,
    s.code,
    s.prefix,
    s.fill,
    s.increment_by,
    s.current_seq
 FROM app_sequence s
 ORDER BY s.id DESC
;

--SELECTED_STORE
SELECT s.id AS store_id FROM app_store s WHERE s.name = %s;

--SELECTED_STORE_BY_ID
SELECT s.name FROM app_store s WHERE s.id = %s;

--INSERT_SALE
INSERT INTO sale
    (amount, sub, discount, tax_amount, delivery_charge, sequence, sequence_type, status, sale_type, login, client_id, store_id, additional_info)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s ) ;

--INSERT_SALE_LINE
INSERT INTO sale_line (amount, tax_amount, discount, quantity, total_amount, sale_id, product_id)
 VALUES (%s, %s, %s, %s, %s, %s, %s)
;

--UPDATE_INV_ON_SALE
UPDATE app_inventory i
  SET  i.prev_quantity = i.quantity ,
       i.quantity = i.quantity - %s
WHERE
      i.product_id = %s
  AND i.store_id = %s
;

--INSERT_USER
INSERT INTO users (username, password, first_name, last_name, is_active, date_joined, pic)
 VALUES(%s, %s, %s, %s, 1, NOW(), %s) RETURNING id
 ;

--INSERT_SCOPES
INSERT INTO scopes (name, user_id)
  VALUES(%s, %s)
;

--INSERT_USER_STORES
INSERT INTO app_user_store (user_id, store_id)
  VALUES (%s, %s)
;

--DELETE_SCOPES
DELETE FROM scopes
 WHERE name = %s
   AND user_id = %s
;

--DELETE_USER_STORES
DELETE FROM app_user_store
  WHERE user_id = %s
    AND store_id = %s
;

--SELECT_SCOPES
SELECT
        s.id,
        s.name
FROM scope_list s
;

--SELECT_DELIVERY
SELECT
    d.id,
    d.name,
    d.value
FROM delivery d
ORDER BY d.id
;


--DASH_BOARD_TOTAL_SALES
SELECT
      SUM(sp.amount) AS total_sale
FROM
     sale s, sale_paid sp
WHERE
	  s.id = sp.sale_id
  AND s.date_create >= DATE_SUB(CURDATE(), INTERVAL 0 DAY)
  AND s.status <> 'RETURN'
;

--DASH_BOARD_TOTAL_SALES_PROMISE
SELECT
      SUM(s.amount) AS total_promise
FROM
     sale s
WHERE
      s.date_create >= DATE_SUB(CURDATE(), INTERVAL 0 DAY)
  AND s.status <> 'RETURN'
;

--DASH_BOARD_TOTAL_INCOME
SELECT
    SUM( s.amount - (
      SELECT
       SUM(l.quantity * p.cost) AS total_cost
		 FROM sale_line l, product p
			WHERE
				 l.sale_id = s.id
			 AND p.id = l.product_id
     )) AS total_income
FROM
    sale s
WHERE
      s.date_create >= DATE_SUB(CURDATE(), INTERVAL 0 DAY)
  AND s.status <> 'RETURN'
;

--INSERT_CLIENT
INSERT INTO client (name, document_id, address, celphone, email, wholesaler)
 VALUES (%s, %s, %s, %s, %s, %s)
;

--FIND_LAST_CLIENT
SELECT id FROM client ORDER BY id DESC LIMIT 1
;
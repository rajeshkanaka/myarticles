## Document for Continuous Email Processing with `flo_scraper`

**Problem:**

We need to configure the `flo_scraper` component to process emails from AppraisalFLO on demand, triggered by the `catchup` component. This will streamline the workflow and eliminate redundant email checking.

**Solution:**

1. **Modify `email_fetcher.py`:**

   - **Remove the infinite loop:** Delete the `while True` loop that was previously used for continuous polling.
   - **Connect and disconnect on demand:** Modify the `email_processing_callback` function to establish the IMAP connection when a message is received from RabbitMQ and disconnect after processing.
   - **Error Handling:** Implement error handling in the `email_processing_callback` function to handle potential RabbitMQ and IMAP connection issues. Use `ch.basic_ack()` to acknowledge messages after successful processing and `ch.basic_nack(requeue=True)` to requeue messages if an error occurs.
   - **Update the command to run in docker-compose file to CMD ["python", "-m", "email_processor"]**.

   **Updated `email_processor.py` Code:**

   ```python
   import imaplib
   # ... (rest of your imports)
   import pika
   import os
   import time

   class EmailFetcher:
       # ... (your existing EmailFetcher code)

   def email_processing_callback(ch, method, properties, body):
       """Callback function triggered by messages from RabbitMQ."""
       global email_fetcher, search_date, search_subject, search_sender
       logger.info(f"Received message: {body.decode()}")

       try:
           email_fetcher.connect()  # Connect to IMAP server
           message_numbers = email_fetcher.search_emails(
               search_date=search_date,
               search_subject=search_subject,
               search_sender=search_sender
           )
           email_fetcher.fetch_and_process_emails(message_numbers)
           ch.basic_ack(delivery_tag=method.delivery_tag)  # Acknowledge the message
       except Exception as e:
           logger.error(f"Error processing emails: {e}")
           ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)  # Requeue the message
       finally:
           email_fetcher.disconnect()  # Disconnect from IMAP server

   def start_processing():
       """Initialize RabbitMQ connection and consume messages."""
       rabbitmq_host = os.getenv('RABBITMQ_HOST')
       rabbitmq_user = os.getenv('RABBITMQ_USER')
       rabbitmq_password = os.getenv('RABBITMQ_PASSWORD')

       credentials = pika.PlainCredentials(rabbitmq_user, rabbitmq_password)
       while True:  # Reconnect loop for resilience
           try:
               connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host, credentials=credentials))
               channel = connection.channel()
               channel.queue_declare(queue='email_processing_queue', durable=True)  # Durable queue
               channel.basic_consume(queue='email_processing_queue', on_message_callback=email_processing_callback, auto_ack=False)
               logger.info('Waiting for email processing messages. To exit press CTRL+C')
               channel.start_consuming()
           except pika.exceptions.AMQPConnectionError as e:
               logger.error(f"RabbitMQ connection error: {e}. Retrying in 5 seconds...")
               time.sleep(5)

   # Example usage
   if __name__ == "__main__":
       # ... (your existing code for loading environment variables and creating EmailFetcher)

       start_processing()  # Start listening for messages
   ```

2. **Modify the `catchup` component:**

   - **Send a message to RabbitMQ:** When the `catchup` component detects new emails that meet the search criteria (date, subject, sender), it should publish a message to the `email_processing_queue` in RabbitMQ.

   **Example Python code for sending a message:**

   ```python
   import pika
   import os

   # ... (your existing code in the catchup component)

   def send_email_processing_trigger():
       """Send a message to RabbitMQ to trigger email processing."""
       rabbitmq_host = os.getenv('RABBITMQ_HOST')
       rabbitmq_user = os.getenv('RABBITMQ_USER')
       rabbitmq_password = os.getenv('RABBITMQ_PASSWORD')

       credentials = pika.PlainCredentials(rabbitmq_user, rabbitmq_password)
       connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host, credentials=credentials))
       channel = connection.channel()

       channel.queue_declare(queue='email_processing_queue', durable=True)

       channel.basic_publish(exchange='',
                             routing_key='email_processing_queue',
                             body='New emails found! Process them now.')

       connection.close()

   # ... (call the send_email_processing_trigger() function in your catchup logic
   #      whenever new emails are found)
   ```

3. **Docker Compose Configuration:**

   - Ensure that the `flo_scraper` container in your `docker-compose.yml` has the correct environment variables for connecting to RabbitMQ.
   - Use `restart: always` to ensure both the `flo_scraper` and RabbitMQ containers restart automatically if they fail.

**Testing:**

- Start the Docker containers using `docker-compose up -d --build`.
- Manually send test emails to the monitored inbox to simulate new email arrivals.
- Observe the RabbitMQ management console to verify message delivery to the queue.
- Check the log files generated by `flo_scraper` to confirm email processing and identify any errors.

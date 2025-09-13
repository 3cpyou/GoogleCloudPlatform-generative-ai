#!/bin/bash
# Deploy Bulk Email Indexer to Cloud Run

echo "🚀 Deploying Bulk Email Indexer to Cloud Run..."

# Copy files to deployment directory
cp bulk_indexer_main.py main.py
cp bulk_indexer_requirements.txt requirements.txt

# Deploy to Cloud Run
gcloud run deploy bulk-email-indexer \
  --source . \
  --region us-central1 \
  --project ad3-sam \
  --memory 4Gi \
  --timeout 3600 \
  --port 8080 \
  --set-env-vars "GOOGLE_CLOUD_PROJECT_ID=ad3-sam,DATABASE_NAME=ad3gem-gmail-lineage,GOOGLE_CLOUD_REGION=us-central1,BATCH_SIZE=100,EMBEDDING_MODEL=text-embedding-004,VECTOR_COLLECTION=email_vectors" \
  --allow-unauthenticated

echo "✅ Deployment complete!"
echo "🌐 Service URL: https://bulk-email-indexer-1008533481658.us-central1.run.app"
echo ""
echo "Usage:"
echo "  # Test with dry run"
echo "  curl -X POST 'https://bulk-email-indexer-1008533481658.us-central1.run.app/index?dry_run=1'"
echo ""
echo "  # Start bulk indexing"
echo "  curl -X POST 'https://bulk-email-indexer-1008533481658.us-central1.run.app/index'"
echo ""
echo "  # Process with custom batch size"
echo "  curl -X POST 'https://bulk-email-indexer-1008533481658.us-central1.run.app/index?batch_size=50'"

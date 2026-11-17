# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Managed Secret Manager container for external travel partner credentials.
# Secret versions are populated out-of-band (`gcloud secrets versions add`) so
# raw credentials never enter Git history or Terraform state.
resource "google_secret_manager_secret" "travel_partner_api_key" {
  project   = var.project_id
  secret_id = "${var.project_name}-partner-api-key"

  replication {
    auto {}
  }

  labels = {
    "managed-by" = "terraform"
    "service"    = var.project_name
  }

  depends_on = [resource.google_project_service.services]
}

resource "google_secret_manager_secret_iam_member" "app_sa_partner_api_key" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.travel_partner_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.app_sa.email}"
}

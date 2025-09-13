# Firestore Collections Structure
*Generated from optimized_gmail_ingestor.py*

This document details the exact JSON structure of all Firestore collections created by the Search-Optimized Gmail Ingestor.

## Collection Overview

The ingestor creates 4 main collections optimized for AI chatbot search patterns:

1. **`email_search_index`** - Primary search layer with first names, search terms
2. **`people_directory`** - Name resolution for natural language queries
3. **`thread_summaries`** - Conversation context and aggregation
4. **`messages_full`** - Complete email content and attachments

---

## 1. email_search_index Collection

**Purpose**: Primary search layer optimized for quick searching and filtering
**Document ID**: Gmail message ID

```json
{
  "messageId": "string",
  "threadId": "string",
  "owner": "string",

  "fromPerson": "string",
  "fromEmail": "string",
  "fromFull": "string",
  "toPeople": ["string"],
  "toEmails": ["string"],
  "allPeople": ["string"],
  "allEmails": ["string"],

  "subject": "string",
  "threadSubject": "string",
  "bodySearchable": "string",
  "searchTerms": ["string"],

  "sentAt": "timestamp",
  "isInternal": "boolean",
  "hasExternal": "boolean",
  "hasAttachments": "boolean",
  "attachmentTypes": ["string"],

  "labelNames": ["string"],
  "isUnread": "boolean",
  "isImportant": "boolean",
  "isStarred": "boolean",

  "snippet": "string",
  "bodyPreview": "string",
  "createdAt": "timestamp"
}
```

### Field Descriptions

| Field             | Type      | Description                           | Format & Example                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------- | --------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `messageId`       | string    | Gmail message ID                      | **Format**: 16-char hex string<br>**Example**: `"18a7374da2f64c3d"`                                                                                                                                                                                                                                                                                                                                                                                                 |
| `threadId`        | string    | Gmail thread ID                       | **Format**: 16-char hex string<br>**Example**: `"18a7374da2f64c3d"`                                                                                                                                                                                                                                                                                                                                                                                                 |
| `owner`           | string    | Email owner (mailbox)                 | **Format**: Lowercase email address<br>**Example**: `"mark@lineagecoffee.com"`                                                                                                                                                                                                                                                                                                                                                                                      |
| `fromPerson`      | string    | Sender's first name                   | **Format**: Lowercase first name extracted from email/display name<br>**Example**: `"coffee"`, `"durban"`, `"jasey"`                                                                                                                                                                                                                                                                                                                                                |
| `fromEmail`       | string    | Sender's email                        | **Format**: Lowercase normalized email address<br>**Example**: `"info@durbanpackaging.co.za"`                                                                                                                                                                                                                                                                                                                                                                       |
| `fromFull`        | string    | Full sender string                    | **Format**: Either bare email OR "Display Name <email@domain.com>"<br>**Examples**: `"coffee@lineagecoffee.com"` OR `"Durban Packaging <info@durbanpackaging.co.za>"`                                                                                                                                                                                                                                                                                               |
| `toPeople`        | array     | Recipient first names                 | **Format**: Array of lowercase strings<br>**Example**: `["coffee"]`, `["tegan"]`, `["itdesk"]`                                                                                                                                                                                                                                                                                                                                                                      |
| `toEmails`        | array     | Recipient emails                      | **Format**: Array of lowercase email addresses<br>**Example**: `["office@lineagecoffee.com"]`                                                                                                                                                                                                                                                                                                                                                                       |
| `allPeople`       | array     | All participants' first names         | **Format**: Array of lowercase first names from all to/from/cc/bcc<br>**Example**: `["craig", "info", "office"]`                                                                                                                                                                                                                                                                                                                                                    |
| `allEmails`       | array     | All participants' emails              | **Format**: Array of lowercase email addresses<br>**Example**: `["craig@lineagecoffee.com", "info@durbanpackaging.co.za"]`                                                                                                                                                                                                                                                                                                                                          |
| `subject`         | string    | Original subject with Re:/Fwd:        | **Format**: Exact original subject line including prefixes<br>**Example**: `"Re: Lineage Coffee - Zapper - [Request ID :##189070##]"`                                                                                                                                                                                                                                                                                                                               |
| `threadSubject`   | string    | Normalized subject (prefixes removed) | **Format**: Subject with "Re:", "Fwd:" prefixes stripped<br>**Example**: `"Lineage Coffee - Zapper - [Request ID :##189070##]"`                                                                                                                                                                                                                                                                                                                                     |
| `bodySearchable`  | string    | First 2000 chars for searching        | **Format**: Subject + "\r\n" + body content + "\r\n" + snippet, truncated to 2000 chars<br>**Example**: `"DBN PACKAGING POP Required Morning Tegan\r\n\r\n \r\n\r\nI hope..."`                                                                                                                                                                                                                                                                                      |
| `searchTerms`     | array     | Extracted keywords + people           | **Format**: Array of lowercase words (3+ chars, no stop words, no duplicates)<br>**Example**: `["email", "kwingin", "fingerprint", "gallery", "30x", "ground"]`                                                                                                                                                                                                                                                                                                     |
| `sentAt`          | timestamp | When sent                             | **Format**: ISO 8601 timestamp with timezone<br>**Example**: `"2023-09-08 06:22:32+00:00"`                                                                                                                                                                                                                                                                                                                                                                          |
| `isInternal`      | boolean   | All participants from company domain  | **Format**: Boolean value<br>**Example**: `true`, `false`                                                                                                                                                                                                                                                                                                                                                                                                           |
| `hasExternal`     | boolean   | Has outside participants              | **Format**: Boolean value<br>**Example**: `true`, `false`                                                                                                                                                                                                                                                                                                                                                                                                           |
| `hasAttachments`  | boolean   | Has attachments                       | **Format**: Boolean value<br>**Example**: `true`, `false`                                                                                                                                                                                                                                                                                                                                                                                                           |
| `attachmentTypes` | array     | Attachment file types                 | **Format**: Array of lowercase file type categories<br>**Example**: `["image"]`, `[]` (empty if no attachments)                                                                                                                                                                                                                                                                                                                                                     |
| `labelNames`      | array     | Gmail label names                     | **Format**: Array of uppercase Gmail label strings<br>**Example**: `["UNREAD", "SENT", "INBOX"]`, `["IMPORTANT", "CATEGORY_PERSONAL", "INBOX"]`                                                                                                                                                                                                                                                                                                                     |
| `isUnread`        | boolean   | UNREAD label present                  | **Format**: Boolean based on "UNREAD" in labelIds<br>**Example**: `true`, `false`                                                                                                                                                                                                                                                                                                                                                                                   |
| `isImportant`     | boolean   | IMPORTANT label present               | **Format**: Boolean based on "IMPORTANT" in labelIds<br>**Example**: `true`, `false`                                                                                                                                                                                                                                                                                                                                                                                |
| `isStarred`       | boolean   | STARRED label present                 | **Format**: Boolean based on "STARRED" in labelIds<br>**Example**: `false`                                                                                                                                                                                                                                                                                                                                                                                          |
| `snippet`         | string    | Gmail snippet preview                 | **Format**: Gmail-generated preview text, may contain escaped HTML<br>**Example**: `"Morning Tegan I hope you are well, Happy Friday, just checking if payment was done please? If so please send POP or advise as to when payment will be made. Thank you, Kind Regards, Emily Lumley From:"`                                                                                                                                                                      |
| `bodyPreview`     | string    | First 300 chars of body               | **Format**: Plain text preview, truncated at 300 chars with "\r\n" line breaks<br>**Example**: `"Morning Tegan\r\n\r\n \r\n\r\nI hope you are well,\r\n\r\n \r\n\r\nHappy Friday, just checking if payment was done please? If so please send POP or advise as to when payment will be made.\r\n\r\n \r\n\r\nThank you,\r\n\r\n \r\n\r\nKind Regards,\r\n\r\nEmily Lumley\r\n\r\nEmail signature B2A - new\r\n\r\n \r\n\r\n \r\n\r\nFrom: Durban Packaging [mailt"` |
| `createdAt`       | timestamp | When ingested                         | **Format**: ISO 8601 timestamp with microseconds and timezone<br>**Example**: `"2025-09-08 19:40:53.027597+00:00"`                                                                                                                                                                                                                                                                                                                                                  |

---

## 2. people_directory Collection

**Purpose**: Name resolution for "julie" → "julie@lineagecoffee.com" lookups
**Document ID**: Normalized email address

```json
{
  "personId": "string",
  "firstName": "string",
  "email": "string",
  "searchKey": "string",
  "isInternal": "boolean",
  "displayName": "string",
  "lastSeen": "timestamp"
}
```

### Field Descriptions

| Field         | Type      | Description                | Format & Example                                                                                                                                              |
| ------------- | --------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `personId`    | string    | Same as email (normalized) | **Format**: Lowercase normalized email address<br>**Example**: `"01000193ac28de4a-37021520-7918-470f-a8b7-e163ca968e5e-000000@amazonses.com"`                 |
| `firstName`   | string    | First name (Title case)    | **Format**: Title Case extracted from email local part or display name<br>**Example**: `"01000193Ac28De4A-37021520-7918-470F-A8B7-E163Ca968E5E-000000"`       |
| `email`       | string    | Normalized email           | **Format**: Lowercase email address (identical to personId)<br>**Example**: `"01000193ac28de4a-37021520-7918-470f-a8b7-e163ca968e5e-000000@amazonses.com"`    |
| `searchKey`   | string    | Lowercase for searching    | **Format**: Lowercase version of firstName for search queries<br>**Example**: `"01000193ac28de4a-37021520-7918-470f-a8b7-e163ca968e5e-000000"`                |
| `isInternal`  | boolean   | From company domain        | **Format**: Boolean - true if email ends with "@lineagecoffee.com"<br>**Example**: `false` (for external emails), `true` (for internal)                       |
| `displayName` | string    | Full display name          | **Format**: Best available display name from email headers or extracted name<br>**Example**: `"01000193Ac28De4A-37021520-7918-470F-A8B7-E163Ca968E5E-000000"` |
| `lastSeen`    | timestamp | Last email activity        | **Format**: ISO 8601 timestamp with microseconds and timezone<br>**Example**: `"2025-09-08 19:39:22.181121+00:00"`                                            |

---

## 3. thread_summaries Collection

**Purpose**: Thread-level overview and conversation aggregation
**Document ID**: Gmail thread ID

```json
{
  "threadId": "string",
  "subject": "string",
  "startDate": "timestamp",
  "lastActivity": "timestamp",
  "messageCount": "number",

  "participants": {
    "email@domain.com": {
      "name": "string",
      "email": "string",
      "isInternal": "boolean"
    }
  },

  "lastMessage": {
    "from": "string",
    "sentAt": "timestamp",
    "preview": "string"
  },

  "keyTopics": ["string"],
  "isInternal": "boolean",
  "hasAttachments": "boolean",
  "updatedAt": "timestamp"
}
```

### Field Descriptions

| Field            | Type      | Description                    | Format & Example                                                                                                                                                                                                               |
| ---------------- | --------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `threadId`       | string    | Gmail thread ID                | **Format**: 16-char hex string<br>**Example**: `"15824df3b52cdedd"`                                                                                                                                                            |
| `subject`        | string    | Thread subject (normalized)    | **Format**: Subject line with Re:/Fwd: prefixes removed<br>**Example**: `"Max on Top"`, `"PRESSBOARD PRO FORMA"`                                                                                                               |
| `startDate`      | timestamp | First message timestamp        | **Format**: ISO 8601 timestamp with timezone<br>**Example**: `"2024-02-05 08:51:54+00:00"`                                                                                                                                     |
| `lastActivity`   | timestamp | Most recent message            | **Format**: ISO 8601 timestamp with timezone<br>**Example**: `"2024-02-05 08:51:54+00:00"`                                                                                                                                     |
| `messageCount`   | number    | Total messages in thread       | **Format**: Integer count<br>**Example**: `1`, `2`, `10`                                                                                                                                                                       |
| `participants`   | object    | MAP of email to person details | **Format**: Object with email addresses as keys, person objects as values<br>**Example**: `{"craig@lineagecoffee.com": {"name": "Craig Charity", "email": "craig@lineagecoffee.com", "isInternal": true}}`                     |
| `lastMessage`    | object    | Latest message summary         | **Format**: Object with from, sentAt, preview fields<br>**Example**: `{"from": "craig@lineagecoffee.com", "sentAt": "2024-02-05 08:51:54+00:00", "preview": "[image: facebook] <https://www.facebook.com/lineagecoffee> ..."}` |
| `keyTopics`      | array     | Top 10 search terms            | **Format**: Array of lowercase extracted keywords (up to 10)<br>**Example**: `["max", "lineage_coffeesa", "shop", "lineage", "pty", "craig"]`                                                                                  |
| `isInternal`     | boolean   | Thread is internal only        | **Format**: Boolean - true if all participants are @lineagecoffee.com<br>**Example**: `true`, `false`                                                                                                                          |
| `hasAttachments` | boolean   | Any message has attachments    | **Format**: Boolean - true if any message in thread has attachments<br>**Example**: `true`, `false`                                                                                                                            |
| `updatedAt`      | timestamp | Last update time               | **Format**: ISO 8601 timestamp with microseconds and timezone<br>**Example**: `"2025-09-08 19:45:56.847225+00:00"`                                                                                                             |

### Participants Structure
**Format**: Object/map where each key is an email address, value is person details object

```json
"participants": {
  "craig@lineagecoffee.com": {
    "name": "Craig Charity",
    "email": "craig@lineagecoffee.com",
    "isInternal": true
  },
  "tanya@natmould.co.za": {
    "name": "Tanya",
    "email": "tanya@natmould.co.za",
    "isInternal": false
  },
  "support@support.xero.com": {
    "name": "Xero Support",
    "email": "support@support.xero.com",
    "isInternal": false
  }
}
```

**Key Format**: Lowercase email address
**Value Fields**:
- `name`: Display name from email headers (Title Case)
- `email`: Lowercase email address (same as key)
- `isInternal`: Boolean - true if email ends with "@lineagecoffee.com"

---

## 4. messages_full Collection

**Purpose**: Complete message storage with full content and metadata
**Document ID**: Gmail message ID

```json
{
  "messageId": "string",
  "threadId": "string",
  "owner": "string",

  "headers": {
    "from": [
      {
        "name": "string",
        "email": "string"
      }
    ],
    "to": [
      {
        "name": "string",
        "email": "string"
      }
    ],
    "cc": [
      {
        "name": "string",
        "email": "string"
      }
    ],
    "bcc": [
      {
        "name": "string",
        "email": "string"
      }
    ],
    "subject": "string",
    "date": "string",
    "messageId": "string",
    "inReplyTo": "string",
    "references": ["string"]
  },

  "bodyText": "string",
  "bodyHtml": "string",
  "snippet": "string",

  "attachments": [
    {
      "filename": "string",
      "mimeType": "string",
      "sizeBytes": "number",
      "gmailAttachmentId": "string"
    }
  ],

  "labelIds": ["string"],
  "labelNames": ["string"],
  "internalDate": "string",
  "historyId": "string",
  "createdAt": "timestamp"
}
```

### Field Descriptions

| Field          | Type      | Description                 | Format & Example                                                                                                                                                                                                                                                      |
| -------------- | --------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `messageId`    | string    | Gmail message ID            | **Format**: 16-char hex string<br>**Example**: `"18a7374da2f64c3d"`                                                                                                                                                                                                   |
| `threadId`     | string    | Gmail thread ID             | **Format**: 16-char hex string<br>**Example**: `"18a7374da2f64c3d"`                                                                                                                                                                                                   |
| `owner`        | string    | Mailbox owner               | **Format**: Lowercase email address of mailbox owner<br>**Example**: `"mark@lineagecoffee.com"`, `"craig@lineagecoffee.com"`                                                                                                                                          |
| `headers`      | object    | Complete email headers      | **Format**: Object with parsed email headers<br>**Example**: See detailed structure below                                                                                                                                                                             |
| `bodyText`     | string    | Complete plain text body    | **Format**: Full email body in plain text with "\r\n" line breaks<br>**Example**: `"Order Date: \n\nOrder from: \nEmail: daniela@kwingin.co.za\nCell: \n\nOrder: \n30x Fingerprint Beans 1kg..."`                                                                     |
| `bodyHtml`     | string    | HTML version (always empty) | **Format**: Always empty string in current implementation<br>**Example**: `""`                                                                                                                                                                                        |
| `snippet`      | string    | Gmail snippet               | **Format**: Gmail-generated preview text<br>**Example**: `"Order Date: Order from: Email: daniela@kwingin.co.za Cell: Order: 30x Fingerprint Beans 1kg 8X250g Decaf Espresso Ground 1x Cleaning Agent Additional information: Please let me know if any amount will"` |
| `attachments`  | array     | Complete attachment details | **Format**: Array of attachment objects<br>**Example**: `[{"mimeType": "image/jpeg", "filename": "image001.jpg", "sizeBytes": 32998, "gmailAttachmentId": "ANGjdJ9Z-z1kpvMHw7kRHSgYWRwk..."}]`                                                                        |
| `labelIds`     | array     | Gmail label IDs             | **Format**: Array of uppercase Gmail label ID strings<br>**Example**: `["UNREAD", "SENT", "INBOX"]`, `["IMPORTANT", "CATEGORY_PERSONAL", "INBOX"]`                                                                                                                    |
| `labelNames`   | array     | Resolved label names        | **Format**: Array of uppercase Gmail label name strings (same as labelIds)<br>**Example**: `["UNREAD", "SENT", "INBOX"]`                                                                                                                                              |
| `internalDate` | string    | Gmail internal timestamp    | **Format**: Unix timestamp in milliseconds as string<br>**Example**: `"1694154152000"`                                                                                                                                                                                |
| `historyId`    | string    | Gmail history ID            | **Format**: Numeric string<br>**Example**: `"749177"`, `"8231967"`                                                                                                                                                                                                    |
| `createdAt`    | timestamp | When ingested               | **Format**: ISO 8601 timestamp with microseconds and timezone<br>**Example**: `"2025-09-08 19:40:53.027615+00:00"`                                                                                                                                                    |

### Headers Structure
**Format**: Object containing parsed email headers with person arrays and metadata

```json
"headers": {
  "from": [
    {
      "email": "info@durbanpackaging.co.za",
      "name": "Durban Packaging"
    }
  ],
  "to": [
    {
      "email": "office@lineagecoffee.com",
      "name": "Tegan Botha"
    }
  ],
  "cc": [
    {
      "email": "craig@lineagecoffee.com",
      "name": ""
    }
  ],
  "bcc": [],
  "subject": "DBN PACKAGING POP Required",
  "date": "2023-09-08T09:25:39+02:00",
  "messageId": "<!&!AAAAAAAAAAAYAAAAAAAAAPdmjfuboc1Ar+vxfdAmbwDCgAAAEAAAANcFz/KtYLtPgrCDT4u1VAgBAAAAAA==@durbanpackaging.co.za>",
  "inReplyTo": "",
  "references": [
    "<CAFtvyM1xp1sepiY3MBkLjNikFC0eHeOL-uHh4Qp8Mv+Ea+whWA@mail.gmail.com>",
    "<!&!AAAAAAAAAAAYAAAAAAAAAPdmjfuboc1Ar+vxfdAmbwDCgAAAEAAAAPaH1IoMY/1HkVd6+0zujnQBAAAAAA==@durbanpackaging.co.za>"
  ]
}
```

**Person Object Format**:
- `email`: Lowercase email address
- `name`: Display name from headers (may be empty string "")

### Attachments Structure
**Format**: Array of attachment objects (empty array if no attachments)

```json
"attachments": [
  {
    "mimeType": "image/jpeg",
    "filename": "image001.jpg",
    "gmailAttachmentId": "ANGjdJ9Z-z1kpvMHw7kRHSgYWRwkwlZWlpl37iMtfB_oi0SxbZ09ud_0_KxgHvryc6ZtBliiLQhlHWr5g7_8z09_TUBDtDHEDG5I6wMQ3QMyKKW63NjLIYvDRbO1_1Df1Y1UQfSgggDyhd3pUXQn1q3BseE0PrrAWgiP928kqtzc1gT6J3K90xW7YiA4ReLMvK21cZfwDNffb9_PYna_zveASg8VukMvt_dQv3RWeOkTS-JuT9RrERO1TnAJHSXUnJ_AVSogVZLPPXET2t2o6o1OWSZ2zxqTOyjagAFDEA21LJ71CmcMVQ-RGnU9yVqvMjWLOUw3xEogepNXg3hDcW3p5VLvEc9WYRVaU4qw8NitJsIWPF28v-jBbxNoTgoz2wmp1B5IyOllO-PGsq4v",
    "sizeBytes": 32998
  }
]
```

**Empty Example** (no attachments):
```json
"attachments": []
```

**Attachment Object Format**:
- `filename`: Original filename from email attachment
- `mimeType`: MIME type (e.g., "image/jpeg", "application/pdf")
- `sizeBytes`: File size in bytes (integer)
- `gmailAttachmentId`: Long Gmail attachment ID string for retrieval

---

## Data Types Reference

| Type        | Description          | Format & Examples                                                                                                                                                                                |
| ----------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `string`    | Text field           | **Format**: UTF-8 text, may contain spaces, special chars, line breaks<br>**Examples**: `"julie@lineagecoffee.com"`, `"DBN PACKAGING POP Required"`, `"Morning Tegan\r\n\r\n \r\n\r\nI hope..."` |
| `array`     | List of values       | **Format**: JSON array with 0 or more elements<br>**Examples**: `["julie", "craig"]`, `[]`, `["UNREAD", "SENT", "INBOX"]`                                                                        |
| `object`    | Nested structure/map | **Format**: JSON object with key-value pairs<br>**Examples**: `{"name": "Julie", "email": "julie@..."}`, `{"from": "craig@...", "sentAt": "2024-02-05..."}`                                      |
| `boolean`   | True/false flag      | **Format**: JSON boolean values<br>**Examples**: `true`, `false`                                                                                                                                 |
| `number`    | Numeric value        | **Format**: Integer values<br>**Examples**: `1`, `2`, `10`, `32998`                                                                                                                              |
| `timestamp` | Firestore timestamp  | **Format**: ISO 8601 string with timezone and optional microseconds<br>**Examples**: `"2023-09-08 06:22:32+00:00"`, `"2025-09-08 19:40:53.027597+00:00"`                                         |

---

## Real-World Examples

### Complete email_search_index Document
**Document ID**: `18a73aec13f59d2a`
**Complete example with all possible fields**:

```json
{
  "toPeople": ["tegan"],
  "messageId": "18a73aec13f59d2a",
  "subject": "DBN PACKAGING POP Required",
  "isImportant": true,
  "snippet": "Morning Tegan I hope you are well, Happy Friday, just checking if payment was done please? If so please send POP or advise as to when payment will be made. Thank you, Kind Regards, Emily Lumley From:",
  "allPeople": ["craig", "info", "office"],
  "isInternal": false,
  "attachmentTypes": ["image"],
  "labelNames": ["IMPORTANT", "CATEGORY_PERSONAL", "INBOX"],
  "allEmails": ["craig@lineagecoffee.com", "info@durbanpackaging.co.za", "office@lineagecoffee.com"],
  "sentAt": "2023-09-08 07:25:39+00:00",
  "bodySearchable": "DBN PACKAGING POP Required Morning Tegan\r\n\r\n \r\n\r\nI hope you are well,\r\n\r\n \r\n\r\nHappy Friday, just checking if payment was done please? If so please send POP or advise as to when payment will be made.\r\n\r\n \r\n\r\nThank you,\r\n\r\n \r\n\r\nKind Regards,\r\n\r\nEmily Lumley\r\n\r\nEmail signature B2A - new\r\n\r\n \r\n\r\n \r\n\r\nFrom: Durban Packaging [mailto:info@durbanpackaging.co.za] \r\nSent: 06 September 2023 10:49 AM\r\nTo: 'Tegan Botha'\r\nSubject: RE: Order for Lineage Coffee\r\n\r\n \r\n\r\nThank you so much\r\n\r\n \r\n\r\nKind Regards,\r\n\r\nEmily Lumley\r\n\r\nEmail signature B2A - new\r\n\r\n \r\n\r\n \r\n\r\nFrom: Tegan Botha [mailto:office@lineagecoffee.com] \r\nSent: 06 September 2023 10:41 AM\r\nTo: Durban Packaging\r\nSubject: Re: Order for Lineage Coffee\r\n\r\n \r\n\r\nHi Emily,\r\n\r\n \r\n\r\nI will chat with Craig and get back to you.\r\n\r\n \r\n\r\nKind regards\r\n\r\nTegan\r\n\r\n \r\n\r\nOn Wed, Sep 6, 2023 at 10:32 AM Durban Packaging <info@durbanpackaging.co.za> wrote:\r\n\r\nHi Tegan\r\n\r\n \r\n\r\nThank you for the order, kindly note the account is still in arrears, we can only deliver once the account is up to date.\r\n\r\n \r\n\r\nPlease can you advise on payment?\r\n\r\n \r\n\r\nKind Regards,\r\n\r\nEmily Lumley\r\n\r\nEmail signature B2A - new\r\n\r\n \r\n\r\n \r\n\r\nFrom: Tegan Botha [mailto:office@lineagecoffee.com] \r\nSent: 06 September 2023 10:15 AM\r\nTo: Durban Packaging\r\nSubject: Order for Lineage Coffee\r\n\r\n \r\n\r\nGood morning,\r\n\r\n \r\n\r\nTrust you are well.\r\n\r\n \r\n\r\nPlease can I place an order for 2x boxes of 500 brown 1kg coffee bags.\r\n\r\n \r\n\r\nThank you\r\n\r\nKind regards\r\n\r\nTegan Morning Tegan I hope you are well, Happy Friday, just checking if payment was done please? If so please send POP or advise as to when payment will be made. Thank you, Kind Regards, Emily Lumley From:",
  "fromPerson": "durban",
  "isUnread": false,
  "searchTerms": ["send", "checking", "payment", "info", "made", "kind", "packaging", "please", "craig", "advise", "well", "office", "pop", "morning", "regards", "friday", "thank", "dbn", "required", "happy", "hope", "tegan", "done"],
  "bodyPreview": "Morning Tegan\r\n\r\n \r\n\r\nI hope you are well,\r\n\r\n \r\n\r\nHappy Friday, just checking if payment was done please? If so please send POP or advise as to when payment will be made.\r\n\r\n \r\n\r\nThank you,\r\n\r\n \r\n\r\nKind Regards,\r\n\r\nEmily Lumley\r\n\r\nEmail signature B2A - new\r\n\r\n \r\n\r\n \r\n\r\nFrom: Durban Packaging [mailt",
  "hasExternal": true,
  "fromFull": "Durban Packaging <info@durbanpackaging.co.za>",
  "fromEmail": "info@durbanpackaging.co.za",
  "createdAt": "2025-09-08 19:48:42.416227+00:00",
  "hasAttachments": true,
  "owner": "craig@lineagecoffee.com",
  "threadId": "18a73aec13f59d2a",
  "isStarred": false,
  "threadSubject": "DBN PACKAGING POP Required",
  "toEmails": ["office@lineagecoffee.com"]
}
```

### Complete people_directory Document
**Document ID**: `01000193ac28de4a-37021520-7918-470f-a8b7-e163ca968e5e-000000@amazonses.com`

```json
{
  "firstName": "01000193Ac28De4A-37021520-7918-470F-A8B7-E163Ca968E5E-000000",
  "displayName": "01000193Ac28De4A-37021520-7918-470F-A8B7-E163Ca968E5E-000000",
  "searchKey": "01000193ac28de4a-37021520-7918-470f-a8b7-e163ca968e5e-000000",
  "lastSeen": "2025-09-08 19:39:22.181121+00:00",
  "isInternal": false,
  "email": "01000193ac28de4a-37021520-7918-470f-a8b7-e163ca968e5e-000000@amazonses.com",
  "personId": "01000193ac28de4a-37021520-7918-470f-a8b7-e163ca968e5e-000000@amazonses.com"
}
```

### Complete thread_summaries Document
**Document ID**: `15f5c6cbecc0ec2e`

```json
{
  "messageCount": 10,
  "subject": "PRESSBOARD PRO FORMA",
  "lastMessage": {
    "from": "craig@lineagecoffee.com",
    "sentAt": "2024-02-14 11:30:58+00:00",
    "preview": "Hi\r\n\r\n\r\n[image: facebook] <https://www.facebook.com/lineagecoffee> \r\n[image: instagram] <https://www.instagram.com/lineage_coffeesa> \r\nCraig Charity\r\n\r\n*Owner*\r\n\r\nLineage Coffee Pty Ltd\r\n   \r\n03103508"
  },
  "lastActivity": "2024-02-14 11:30:58+00:00",
  "isInternal": true,
  "participants": {
    "tanya@natmould.co.za": {
      "name": "Tanya",
      "email": "tanya@natmould.co.za",
      "isInternal": false
    },
    "craigcharity@lineagecoffee.com": {
      "name": "Craig Charity",
      "email": "craigcharity@lineagecoffee.com",
      "isInternal": true
    },
    "craig@lineagecoffee.com": {
      "name": "Craig Charity",
      "email": "craig@lineagecoffee.com",
      "isInternal": true
    }
  },
  "hasAttachments": false,
  "keyTopics": ["lineage_coffeesa", "shop", "pressboard", "lineage", "pty", "craig", "lineagecoffee", "craigcharity", "owner", "charity"],
  "threadId": "15f5c6cbecc0ec2e",
  "startDate": "2024-02-14 11:30:58+00:00",
  "updatedAt": "2025-09-08 19:45:42.848312+00:00"
}
```

### Complete messages_full Document
**Document ID**: `18a73aec13f59d2a` (with attachments and complete headers)

```json
{
  "messageId": "18a73aec13f59d2a",
  "internalDate": "1694159373000",
  "historyId": "1854783",
  "labelIds": ["UNREAD", "IMPORTANT", "CATEGORY_PERSONAL", "INBOX"],
  "snippet": "Morning Tegan I hope you are well, Happy Friday, just checking if payment was done please? If so please send POP or advise as to when payment will be made. Thank you, Kind Regards, Emily Lumley From:",
  "attachments": [
    {
      "mimeType": "image/jpeg",
      "filename": "image001.jpg",
      "gmailAttachmentId": "ANGjdJ9Z-z1kpvMHw7kRHSgYWRwkwlZWlpl37iMtfB_oi0SxbZ09ud_0_KxgHvryc6ZtBliiLQhlHWr5g7_8z09_TUBDtDHEDG5I6wMQ3QMyKKW63NjLIYvDRbO1_1Df1Y1UQfSgggDyhd3pUXQn1q3BseE0PrrAWgiP928kqtzc1gT6J3K90xW7YiA4ReLMvK21cZfwDNffb9_PYna_zveASg8VukMvt_dQv3RWeOkTS-JuT9RrERO1TnAJHSXUnJ_AVSogVZLPPXET2t2o6o1OWSZ2zxqTOyjagAFDEA21LJ71CmcMVQ-RGnU9yVqvMjWLOUw3xEogepNXg3hDcW3p5VLvEc9WYRVaU4qw8NitJsIWPF28v-jBbxNoTgoz2wmp1B5IyOllO-PGsq4v",
      "sizeBytes": 32998
    }
  ],
  "owner": "craig@lineagecoffee.com",
  "headers": {
    "messageId": "<!&!AAAAAAAAAAAYAAAAAAAAAPdmjfuboc1Ar+vxfdAmbwDCgAAAEAAAANcFz/KtYLtPgrCDT4u1VAgBAAAAAA==@durbanpackaging.co.za>",
    "date": "2023-09-08T09:25:39+02:00",
    "references": [
      "<CAFtvyM1xp1sepiY3MBkLjNikFC0eHeOL-uHh4Qp8Mv+Ea+whWA@mail.gmail.com>",
      "<!&!AAAAAAAAAAAYAAAAAAAAAPdmjfuboc1Ar+vxfdAmbwDCgAAAEAAAAPaH1IoMY/1HkVd6+0zujnQBAAAAAA==@durbanpackaging.co.za>",
      "<CAFtvyM2GrCBM-usToLc7y3_=NSiavP=1c+19mS6Hu4cymNmZsw@mail.gmail.com>"
    ],
    "subject": "DBN PACKAGING POP Required",
    "inReplyTo": "",
    "from": [
      {
        "email": "info@durbanpackaging.co.za",
        "name": "Durban Packaging"
      }
    ],
    "bcc": [],
    "to": [
      {
        "email": "office@lineagecoffee.com",
        "name": "Tegan Botha"
      }
    ],
    "cc": [
      {
        "email": "craig@lineagecoffee.com",
        "name": ""
      }
    ]
  },
  "createdAt": "2025-09-08 19:48:42.416247+00:00",
  "labelNames": ["IMPORTANT", "CATEGORY_PERSONAL", "INBOX"],
  "bodyHtml": "",
  "threadId": "18a73aec13f59d2a",
  "bodyText": "Morning Tegan\r\n\r\n \r\n\r\nI hope you are well,\r\n\r\n \r\n\r\nHappy Friday, just checking if payment was done please? If so please send POP or advise as to when payment will be made.\r\n\r\n \r\n\r\nThank you,\r\n\r\n \r\n\r\nKind Regards,\r\n\r\nEmily Lumley\r\n\r\nEmail signature B2A - new\r\n\r\n \r\n\r\n \r\n\r\nFrom: Durban Packaging [mailto:info@durbanpackaging.co.za] \r\nSent: 06 September 2023 10:49 AM\r\nTo: 'Tegan Botha'\r\nSubject: RE: Order for Lineage Coffee\r\n\r\n \r\n\r\nThank you so much\r\n\r\n \r\n\r\nKind Regards,\r\n\r\nEmily Lumley\r\n\r\nEmail signature B2A - new\r\n\r\n \r\n\r\n \r\n\r\nFrom: Tegan Botha [mailto:office@lineagecoffee.com] \r\nSent: 06 September 2023 10:41 AM\r\nTo: Durban Packaging\r\nSubject: Re: Order for Lineage Coffee\r\n\r\n \r\n\r\nHi Emily,\r\n\r\n \r\n\r\nI will chat with Craig and get back to you.\r\n\r\n \r\n\r\nKind regards\r\n\r\nTegan\r\n\r\n \r\n\r\nOn Wed, Sep 6, 2023 at 10:32 AM Durban Packaging <info@durbanpackaging.co.za> wrote:\r\n\r\nHi Tegan\r\n\r\n \r\n\r\nThank you for the order, kindly note the account is still in arrears, we can only deliver once the account is up to date.\r\n\r\n \r\n\r\nPlease can you advise on payment?\r\n\r\n \r\n\r\nKind Regards,\r\n\r\nEmily Lumley\r\n\r\nEmail signature B2A - new\r\n\r\n \r\n\r\n \r\n\r\nFrom: Tegan Botha [mailto:office@lineagecoffee.com] \r\nSent: 06 September 2023 10:15 AM\r\nTo: Durban Packaging\r\nSubject: Order for Lineage Coffee\r\n\r\n \r\n\r\nGood morning,\r\n\r\n \r\n\r\nTrust you are well.\r\n\r\n \r\n\r\nPlease can I place an order for 2x boxes of 500 brown 1kg coffee bags.\r\n\r\n \r\n\r\nThank you\r\n\r\nKind regards\r\n\r\nTegan"
}
```

---

## Key Design Features

### Search Optimization
- **First names extracted** for natural language queries ("julie" → "julie@lineagecoffee.com")
- **Search terms** extracted from content for keyword matching
- **Multiple people arrays** for different participant query patterns

### Denormalization Strategy
- Same data stored multiple ways for query optimization
- Flat structure in `email_search_index` for fast filtering
- Rich context in `thread_summaries` for conversation understanding

### Participant Handling
- **Participants as MAP** in thread_summaries (not array)
- **All participants tracked** across from/to/cc/bcc
- **Internal/external classification** for privacy filtering

### Content Search
- **Body searchable** field limited to 2000 chars for indexing
- **Search terms** automatically extracted and include people names
- **Key topics** derived from search terms for thread categorization

### Important Notes
- **`bodyHtml` field**: Always empty string `""` in current implementation. HTML content is extracted and converted to plain text using html2text, then stored in `bodyText`
- **Participants structure**: Stored as MAP/object in thread_summaries (not array), which affects query patterns
- **Search optimization**: Multiple denormalized fields for different query scenarios
- **Line breaks**: Text fields use `\r\n` (CRLF) for line breaks in body content
- **Email normalization**: All email addresses stored in lowercase throughout all collections
- **Name extraction**: First names extracted from email local parts or display names, stored in lowercase for search
- **Timestamp format**: Consistent ISO 8601 format with timezone, microseconds included in `createdAt` and `updatedAt` fields
- **Empty arrays**: Fields that can be arrays are never null, always `[]` when empty
- **Gmail IDs**: Message and thread IDs are 16-character hexadecimal strings from Gmail API

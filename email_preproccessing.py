def getcharsets(msg):
    charsets = set({})
    for c in msg.get_charsets():
        if c is not None:
            charsets.update([c])
    return charsets

def handleerror(errmsg, emailmsg,cs):
    print()
    print(errmsg)
    print("This error occurred while decoding with ",cs," charset.")
    print("These charsets were found in the one email.",getcharsets(emailmsg))
    print("This is the subject:",emailmsg['subject'])
    print("This is the sender:",emailmsg['From'])
    
def getbodyfromemail(msg):
    body = None
    #Walk through the parts of the email to find the text body.    
    if msg.is_multipart():    
        for part in msg.walk():

            # If part is multipart, walk through the subparts.            
            if part.is_multipart(): 

                for subpart in part.walk():
                    if subpart.get_content_type() == 'text/plain':
                        # Get the subpart payload (i.e the message body)
                        body = subpart.get_payload(decode=True) 
                        #charset = subpart.get_charset()

            # Part isn't multipart so get the email body
            elif part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)
                #charset = part.get_charset()

    # If this isn't a multi-part message then get the payload (i.e the message body)
    elif msg.get_content_type() == 'text/plain':
        body = msg.get_payload(decode=True) 

   # No checking done to match the charset with the correct part. 
#     for charset in getcharsets(msg):
#         try:
#             body = body.decode(charset)
#         except UnicodeDecodeError:
#             handleerror("UnicodeDecodeError: encountered.",msg,charset)
#         except AttributeError:
#              handleerror("AttributeError: encountered" ,msg,charset)
    return body    


def parse_emails(mbox_messages):
    email_bodies = []
    email_subjects = []
    email_ids = []
    email_content_types = []
    for i, message in enumerate(mbox_messages):
        body = getbodyfromemail(message)
        email_bodies.append(body)
#         body = message.get_payload()
#         if isinstance(body, str):
#             email_bodies.append(body)
#         else:
#             new_body = ' '.join(body)
#             email_bodies.append(new_body)
        if message['Subject']:
            email_subjects.append(message['Subject'])
        else:
            email_subjects.append("Empty")
        email_ids.append(message['X-UID'])
        if message['Content-Type']:
            email_content_types.append(message['Content-Type'])
        else:
            email_content_types.append("Empty")
    return email_bodies, email_subjects, email_ids, email_content_types


def del_punct_symbols(texts):
    texts = [str(text).lower().replace('\n',' ').replace('\t',' ') for text in texts]
    texts = [re.sub(r'[^\w\s]','',str(text)) for text in texts]
    return texts


def del_stop_words(texts, stop_words):
    return [[word for word in email.split() if word not in stop_words] for email in texts]


def lemmatize_text(texts, lemmatizer):
    return [[lemmatizer.lemmatize(word) for word in email] for email in texts]
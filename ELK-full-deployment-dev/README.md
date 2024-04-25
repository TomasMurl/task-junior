# # # ELK-full-deployment
Configurations for ELK deployment

# 1. Generate certificate authority
elasticsearch-certutil ca --pem

# 2. Generate certificates for each node and agents
elasticsearch-certutil cert --name client -dns --ca-cert /path/to/ca/ca.crt --ca-key /path/to/ca/ca.key  --pem
Mount corresponding certificates to containers

# 3. Generate encryption keys  
kibana-encryption-keys generate

# 4. Generate passwords
elasticsearch-setup-passwords auto
elasticsearch-reset-password --user {user} --url {https://elastic.local:9200} 

# 5. Turn on Fleet
Kibana UI -> Managment -> Fleet -> policy

# 6. Load dashboards from agents into Kibana
ex. sudo ./auditbeat --setup --dashboards -e

# 7. Start agents
ex. sudo ./auditbeat -c auditbeat.yml -e -v

# 8. Create view
Kibana -> Analytics -> Discover -> Create new data view

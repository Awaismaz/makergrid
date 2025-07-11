# Generated by Django 5.2.4 on 2025-07-02 21:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0003_remove_customuser_subscription_end_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='PendingSignup',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('email', models.EmailField(max_length=254, unique=True)),
                ('username', models.CharField(max_length=150)),
                ('full_name', models.CharField(blank=True, max_length=150)),
                ('organization', models.CharField(blank=True, max_length=150)),
                ('password', models.CharField(max_length=128)),
                ('otp', models.CharField(max_length=6)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('expires_at', models.DateTimeField()),
            ],
        ),
    ]

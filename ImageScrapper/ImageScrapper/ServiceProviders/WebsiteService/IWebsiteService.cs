namespace ImageScrapper.ServiceProviders.WebsiteService;

public interface IWebsiteService
{
    Task<IList<object>> GetProductList(string websiteUrl);
    Task StartScrapping();
}